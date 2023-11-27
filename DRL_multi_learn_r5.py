import math
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Input
from collections import deque
import time
from DRL_env import DRLenv
import gc

from memory_profiler import profile


class DRLmultiagent(object):
    def __init__(self, state_size, action_size, action_cand, pmax, noise):

        self.initial_learning_rate = 5e-3
        self.learning_rate = self.initial_learning_rate
        self.lambda_lr = 1e-4  # decay rate for learning rate

        self.gamma = 0.5

        self.pmax = pmax #38dbm

        self.state_size = state_size
        self.action_size = action_size
        self.action_cand = action_cand
        self.action_set = np.linspace(0, self.pmax, self.action_cand)
        #self.action_set = np.logspace(0, math.log(self.pmax), self.action_cand)
        self.transmitters = 19
        self.users = 19

        self.env = DRLenv()
        self.A = self.env.tx_positions_gen(self.transmitters, 100)
        self.B, self.inside_status = self.env.rx_positions_gen(self.A, 10, 100)

        self.noise = noise

        self.model = self.build_network

        self.replay_buffer = deque(maxlen=self.transmitters*1000)
        self.update_rate = 100
        self.main_network = self.build_network()
        weight = self.main_network.get_weights()

        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())

        self.loss = []

        self.temp_reward1 = 0

        self.c = 5

    def update_learning_rate(self):
        self.learning_rate *= (1 - self.lambda_lr)

        # Update learning rate for the main network
        self.main_network.optimizer.lr.assign(self.learning_rate)

        # Update learning rate for the single target network
        self.target_network.optimizer.lr.assign(self.learning_rate)


    def build_network(self):
        model = Sequential()
        model.add(Dense(200, activation="tanh", input_shape=(self.state_size,)))
        model.add(Dense(100, activation="tanh"))
        model.add(Dense(40, activation="tanh"))
        model.add(Dense(self.action_size, activation="tanh"))
        #model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        #sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        model.compile(loss="mse", optimizer=sgd_optimizer)
        #model.summary()
        return model

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state, epsilon):
        if np.random.random() <= epsilon:
            action_temp = np.random.choice(len(self.action_set))
            action = self.action_set[int(action_temp)]
            print('EPS power: ', action)
        else:
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            Q_values = self.target_network.predict(state_tensor, verbose=0)
            action_temp = np.argmax(Q_values[0])
            action = self.action_set[int(action_temp)]
            print('GRD power: ', action)

        return action

    def compute_sum_rate(self, channel_gain, actions, noise):
        sum_rate = 0
        for i in range(len(actions)):
            interferences = sum(channel_gain[j, i] * actions[j] for j in range(len(actions)) if j != i)
            sum_rate += math.log2(1 + channel_gain[i, i] * actions[i] / (interferences + noise))
        return sum_rate

    def sort_and_select_top_c(self, agent, channel_gain, next_channel_gain, actions, c):
        interferer_gain = [(j, next_channel_gain[j, agent] * actions[agent]) for j in range(self.transmitters) if j != agent]
        interfered_gain = [(k, (channel_gain[agent, k] * actions[agent]) / (np.sum(channel_gain[:, k] * actions) - channel_gain[k, k] * actions[k] + self.noise)) for k in range(self.transmitters) if k != agent]

        # Sort based on gain and select top c
        top_c_interferers = sorted(interferer_gain, key=lambda x: x[1], reverse=True)[:c]
        top_c_interfered = sorted(interfered_gain, key=lambda x: x[1], reverse=True)[:c]

        return top_c_interferers, top_c_interfered




    def step(self, state, actions, TTI, max_TTI, channel_gain, next_channel_gain, agent):

        if TTI >= max_TTI:
            done = True
        else:
            done = False

        if hasattr(self, 'next_state'):
            self.next_state.fill(0)
        else:
            self.next_state = np.zeros([self.state_size])

        top_c_interferers, top_c_interfered = self.sort_and_select_top_c(agent, channel_gain, next_channel_gain, actions, self.c)

        direct_signal = channel_gain[agent, agent] * actions[agent]
        inter = np.sum(channel_gain[:, agent] * actions) - direct_signal

        self.temp_reward1 = math.log2(1 + direct_signal / (inter + self.noise))

        reward = self.temp_reward1
        for j, _ in top_c_interfered:
            inter_of_interfered = np.sum(channel_gain[:, j] * actions) - channel_gain[j, j] * actions[j]
            rate_with_agent = math.log2(1 + channel_gain[j, j] * actions[j] / (inter_of_interfered + self.noise))
            inter_of_interfered_without_agent = inter_of_interfered - channel_gain[agent, j] * actions[agent]
            rate_without_agent = math.log2(
                1 + channel_gain[j, j] * actions[j] / (inter_of_interfered_without_agent + self.noise))
            reward -= (rate_without_agent - rate_with_agent)


        self.next_state[0] = actions[agent]
        self.next_state[1] = self.temp_reward1
        self.next_state[2] = next_channel_gain[agent, agent]
        self.next_state[3] = channel_gain[agent, agent]
        self.next_state[4] = np.sum(next_channel_gain[:, agent] * actions) - next_channel_gain[agent, agent] * actions[agent] + self.noise
        self.next_state[5] = state[4]

        state_index = 6

        for j, _ in top_c_interferers:
            # Calculate interference from other agents
            # ...

            # Update the state array with interferer information
            self.next_state[state_index] = next_channel_gain[j, agent] * actions[j]
            self.next_state[state_index + 1] = math.log2(1 + channel_gain[j, j] * actions[j] / (
                        np.sum(channel_gain[:, j] * actions) - channel_gain[j, j] * actions[j] + self.noise))

            self.next_state[state_index + 2] = state[state_index]

            self.next_state[state_index + 3] = state[state_index + 1]

            # Move to the next set of indices for the next interferer
            state_index += 4

        for k, _ in top_c_interfered:
            self.next_state[state_index] = channel_gain[k, k]

            self.next_state[state_index + 1] = math.log2(1 + channel_gain[k, k] * actions[k] / (
                        np.sum(channel_gain[:, k] * actions) - channel_gain[k, k] * actions[k] + self.noise))

            self.next_state[state_index + 2] = (channel_gain[agent, k] * actions[agent]) / (
                        np.sum(channel_gain[:, k] * actions) - channel_gain[k, k] * actions[k] + self.noise)

            state_index += 3

        gc.collect()

        info = {}

        return self.next_state, reward, done, info

    def train(self, batch_size):
        self.update_learning_rate()

        # Efficiently sample a minibatch
        minibatch = random.sample(self.replay_buffer, batch_size)

        # Separate the data into batches
        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        # Convert to tensors in one go
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Compute the Q value using the target network for all next states
        future_qs = np.amax(self.target_network.predict(next_states_tensor, verbose=0), axis=1)
        target_qs = rewards_tensor + self.gamma * future_qs * (1 - dones_tensor)

        # Get the current Q values and update them
        current_qs = self.main_network.predict(states_tensor, verbose=0)
        actions_indices = np.array([np.where(self.action_set == action)[0][0] for action in actions])
        current_qs[np.arange(batch_size), actions_indices] = target_qs

        # Train the main network in one go
        result = self.main_network.fit(states_tensor, current_qs, epochs=1, verbose=0)

        # Append the average loss of this batch to the loss list
        self.loss.append(np.mean(result.history['loss']))

    def update_target_network(self):
        weight = self.main_network.get_weights()
        self.target_network.set_weights(weight)


