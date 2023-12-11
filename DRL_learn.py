import math
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
import random
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import Input
from collections import deque
import time
from DRL_env import DRLenv
import gc

from memory_profiler import profile


class DRLagent(object):
    def __init__(self, state_size, action_size, action_cand, pmax, noise):

        self.initial_learning_rate = 5e-3
        self.learning_rate = self.initial_learning_rate
        self.lambda_lr = 1e-4  # decay rate for learning rate

        self.gamma = 0.9

        self.pmax = pmax #38dbm

        self.state_size = state_size
        self.action_size = action_size
        self.action_cand = action_cand
        self.action_set = np.linspace(0, self.pmax, self.action_cand)

        self.transmitters = 3
        self.users = 3

        self.env = DRLenv()
        #self.A = self.env.tx_positions_gen()
        #self.B = self.env.rx_positions_gen(self.A)
        self.A = self.env.tx_positions_gen(self.transmitters, 100)
        self.B, self.inside_status = self.env.rx_positions_gen(self.A, 10, 100)

        self.noise = noise

        #self.model = self.build_network

        self.replay_buffer = deque(maxlen=100)
        self.update_rate = 100
        self.main_network = self.build_network_main()

        self.target_network = self.build_network_target()
        self.target_network.set_weights(self.main_network.get_weights())

        self.loss = []

        self.temp_reward1 = 0

        self.SINR_cap = 10 ** (30 / 10)

    def update_learning_rate(self):
        self.learning_rate *= (1 - self.lambda_lr)

        # Update learning rate for the main network
        self.main_network.optimizer.lr.assign(self.learning_rate)

        # Update learning rate for the single target network
        #self.target_network.optimizer.lr.assign(self.learning_rate)


    def build_network_main(self):
        model = Sequential()
        model.add(Dense(200, activation="tanh", input_shape=(self.state_size,)))
        model.add(Dense(100, activation="tanh"))
        model.add(Dense(40, activation="tanh"))
        model.add(Dense(self.action_size, activation="tanh"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        #sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        #sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        #model.compile(loss="mse", optimizer=sgd_optimizer)
        #model.summary()
        return model

    def build_network_target(self):
        model = Sequential()
        model.add(Dense(200, activation="tanh", input_shape=(self.state_size,)))
        model.add(Dense(100, activation="tanh"))
        model.add(Dense(40, activation="tanh"))
        model.add(Dense(self.action_size, activation="tanh"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.initial_learning_rate))
        #sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9)
        #sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        #model.compile(loss="mse", optimizer=sgd_optimizer)
        #model.summary()
        return model


    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state, epsilon):
        actions = np.zeros(self.transmitters)
        if np.random.random() <= epsilon:
            # Exploration: Randomly choose an action for each transmitter
            for i in range(self.transmitters):
                action_temp = np.random.choice(len(self.action_set))
                actions[i] = self.action_set[int(action_temp)]
        else:
            # Exploitation: Choose the best action based on Q-values
            state_tensor = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
            Q_values = self.main_network.predict(state_tensor, verbose=0)

            # Reshape Q_values to match the action space structure
            reshaped_Q_values = Q_values.reshape(-1, len(self.action_set))

            # Select the best action for each transmitter
            for i in range(self.transmitters):
                best_action_index = np.argmax(reshaped_Q_values[i])
                actions[i] = self.action_set[best_action_index]

        return actions
    def compute_sum_rate(self, channel_gain, actions, noise):
        sum_rate = 0
        SINR_cap = 10 ** (30 / 10)
        for i in range(len(actions)):
            interferences = sum(channel_gain[j, i] * actions[j] for j in range(len(actions)) if j != i)
            SINR = channel_gain[i, i] * actions[i] / (interferences + noise)

            # Cap the SINR at 30 dB
            SINR = min(SINR, SINR_cap)
            # if SINR == SINR_cap:
            # print('It is over 30dB')

            sum_rate += math.log(1 + SINR)
        return sum_rate

    def normalize(self, value, max_value):
        return value / max_value

    def step(self, state, actions, TTI, max_TTI, channel_gain, next_channel_gain):

        if TTI >= max_TTI:
            done = True
        else:
            done = False

        if hasattr(self, 'next_state'):
            self.next_state.fill(0)
        else:
            self.next_state = np.zeros([self.state_size])

        reward = self.compute_sum_rate(channel_gain, actions, self.noise)
        '''
        for j, _ in top_c_interfered:
            inter_of_interfered = np.sum(channel_gain[:, j] * actions) - channel_gain[j, j] * actions[j]
            rate_with_agent = math.log10(1 + channel_gain[j, j] * actions[j] / (inter_of_interfered + self.noise))
            inter_of_interfered_without_agent = inter_of_interfered - channel_gain[agent, j] * actions[agent]
            rate_without_agent = math.log10(
                1 + channel_gain[j, j] * actions[j] / (inter_of_interfered_without_agent + self.noise))
            reward -= (rate_without_agent - rate_with_agent)
        '''

        state_index = 0

        for i in range(self.transmitters):
            for j in range(self.transmitters):
                if j != i:
                    # Calculate interference from other agents
                    # ...

                    # Update the state array with interferer information
                    self.next_state[state_index] = actions[j]
                    self.next_state[state_index + 1] = next_channel_gain[j, i] * actions[j]
                    self.next_state[state_index+2] = state[state_index]
                    self.next_state[state_index + 3] = state[state_index + 1]
                    # Move to the next set of indices for the next interferer
                    state_index += 2

        gc.collect()

        info = {}

        return self.next_state, reward, done, info

    def action_vectors_to_indices(self, action_vectors):
        """
        Convert action vectors to indices in the Q-value matrix.
        """
        indices = []
        for action_vector in action_vectors:
            index = 0
            for i, action in enumerate(action_vector):
                index += np.where(self.action_set == action)[0][0] * (len(self.action_set) ** i)
            indices.append(index)
        return np.array(indices)

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

        future_qs = np.amax(self.target_network.predict(next_states_tensor, verbose=0), axis=1)
        target_qs = rewards_tensor + self.gamma * future_qs * (1 - dones_tensor)

        # Get the current Q values and update them
        current_qs = self.main_network.predict(states_tensor, verbose=0)

        # Convert action vectors to indices
        action_indices = self.action_vectors_to_indices(actions)

        # Update the Q values for the taken actions
        current_qs[np.arange(batch_size), action_indices] = target_qs

        # Train the main network in one go
        result = self.main_network.fit(states_tensor, current_qs, epochs=1, verbose=1)

        # Append the average loss of this batch to the loss list
        self.loss.append(np.mean(result.history['loss']))

    def update_target_network(self):
        weight = self.main_network.get_weights()
        self.target_network.set_weights(weight)

    def soft_update_target_network(self, tau):
        main_weights = self.main_network.get_weights()
        target_weights = self.target_network.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = (1 - tau) * target_weights[i] + tau * main_weights[i]

        self.target_network.set_weights(target_weights)



