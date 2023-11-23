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
from tensorflow.keras.layers import Input
from collections import deque
import time
from DRL_env import DRLenv
import gc

from memory_profiler import profile


class DRLmultiagent(object):
    def __init__(self, state_size, action_size, action_cand, pmax, noise):

        self.initial_learning_rate = 1e-2
        self.learning_rate = self.initial_learning_rate
        self.lambda_lr = 1e-4  # decay rate for learning rate
        #self.learning_rate_decay = 1-math.pow(10,-4)

        self.gamma = 0.5

        self.pmax = pmax #38dbm

        self.state_size = state_size
        self.action_size = action_size
        self.action_cand = action_cand
        self.action_set = np.linspace(0, self.pmax, self.action_cand)


        #self.action = np.zeros(action_size)

        self.transmitters = 3
        self.users = 3

        self.env = DRLenv()
        self.A = self.env.tx_positions_gen()
        self.B = self.env.rx_positions_gen(self.A)

        self.noise = noise

        self.model = self.build_network

        self.replay_buffer = deque(maxlen=self.transmitters*1000)
        self.update_rate = 100
        self.main_network = self.build_network()
        weight = self.main_network.get_weights()

        for i in range(1, self.transmitters + 1):
            setattr(self, f'target_network{i}', self.build_network())
            getattr(self, f'target_network{i}').set_weights(weight)

        self.loss = []

        self.temp_reward1 = 0

    def update_learning_rate(self):
        self.learning_rate *= (1 - self.lambda_lr)
        for i in range(1, self.transmitters + 1):
            getattr(self, f'target_network{i}').optimizer.lr.assign(self.learning_rate)
        self.main_network.optimizer.lr.assign(self.learning_rate)



    def build_network(self):
        model = Sequential()
        model.add(Dense(200, activation="tanh", input_shape=(self.state_size,)))
        model.add(Dense(100, activation="tanh"))
        model.add(Dense(40, activation="tanh"))
        model.add(Dense(self.action_size, activation="tanh"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        model.summary()
        return model

    def store_transistion(self, state, action, reward, next_state, done, agent):
        self.replay_buffer.append((state, action, reward, next_state, done, agent))

    def epsilon_greedy(self, agent, state, epsilon):
        if np.random.random() <= epsilon:
            action_temp = np.random.choice(len(self.action_set))
            action = self.action_set[int(action_temp)]
            print('EPS agent: ', agent, 'power: ', action)

        else:
            Q_values = getattr(self, f'target_network{agent + 1}').predict(state.reshape(1, -1))
            #Q_values = self.main_network.predict(state.reshape(1, -1))
            action_temp = np.argmax(Q_values[0])
            action = self.action_set[int(action_temp)]
            print('GRD agent: ', agent, 'power: ', action)

        return action


    def step(self, state, actions, TTI, max_TTI, channel_gain, next_channel_gain, agent):

        if TTI >= max_TTI:
            done = True
        else:
            done = False

        if hasattr(self, 'next_state'):
            self.next_state.fill(0)
        else:
            self.next_state = np.zeros([self.state_size])

        direct_signal = channel_gain[agent, agent] * actions[agent]
        inter = np.sum(channel_gain[:, agent] * actions) - direct_signal

        self.temp_reward1 = math.log2(1 + direct_signal / (inter + self.noise))

        reward = self.temp_reward1
        for j in range(self.users):
            if j != agent:
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
        self.next_state[4] = np.sum(next_channel_gain[:, agent] * actions) - next_channel_gain[agent, agent] * actions[agent]
        self.next_state[5] = state[4]

        state_index = 6

        for j in range(self.transmitters):
            if j != agent:
                # Calculate interference from other agents
                # ...

                # Update the state array with interferer information
                self.next_state[state_index] = next_channel_gain[j, agent] * actions[j]
                self.next_state[state_index + 1] = math.log2(1 + channel_gain[j, j] * actions[j] / (np.sum(channel_gain[:, j] * actions) - channel_gain[j, j] * actions[j] + self.noise))

                self.next_state[state_index + 2] = state[state_index]

                self.next_state[state_index + 3] = state[state_index+1]

                # Move to the next set of indices for the next interferer
                state_index += 4

        for k in range(self.transmitters):
            if k != agent:

                self.next_state[state_index] = channel_gain[k, k]

                self.next_state[state_index + 1] = math.log2(1 + channel_gain[k, k] * actions[k] / (np.sum(channel_gain[:, k] * actions) - channel_gain[k, k] * actions[k] + self.noise))

                self.next_state[state_index + 2] = (channel_gain[agent, k] * actions[agent]) / (np.sum(channel_gain[:, k] * actions) - channel_gain[k, k] * actions[k] + self.noise)

                state_index += 3

        gc.collect()

        info = {}

        return self.next_state, reward, done, info

    def train(self, batch_size):

        self.update_learning_rate()

        # compute the Q value using the target network
        minibatch = random.sample(self.replay_buffer, batch_size)
        '''
        for state, action, reward, next_state, done, agent in minibatch:
            target_Q = reward
            if not done:
                max_future_q = np.amax(
                    getattr(self, f'target_network{agent + 1}').predict(next_state.reshape(1, -1))[0])
                target_Q += self.gamma * max_future_q

            current_qs = self.main_network.predict(state.reshape(1, -1))
            action_index = np.where(self.action_set == action)[0][0]
            current_qs[0][action_index] = target_Q

            # train the main network
            result = self.main_network.fit(state.reshape(1, -1), current_qs, epochs=1, verbose=1)

            self.loss.append(result.history['loss'])
        '''

        states = np.zeros((batch_size, self.state_size))
        target_qs_batch = np.zeros((batch_size, self.action_size))

        # Loop over the minibatch and compute target Q values
        for i, (state, action, reward, next_state, done, agent) in enumerate(minibatch):
            target_Q = reward
            if not done:
                max_future_q = np.amax(
                    getattr(self, f'target_network{agent + 1}').predict(next_state.reshape(1, -1))[0])
                target_Q += self.gamma * max_future_q

            current_qs = self.main_network.predict(state.reshape(1, -1))[0]
            action_index = np.where(self.action_set == action)[0][0]
            current_qs[action_index] = target_Q

            # Store the state and the target Q values in their respective arrays
            states[i] = state
            target_qs_batch[i] = current_qs

        # Train the main network in one go
        result = self.main_network.fit(states, target_qs_batch, epochs=1, verbose=1)

        # Append the average loss of this batch to the loss list
        self.loss.append(np.mean(result.history['loss']))




    def update_target_network(self):
        weight = self.main_network.get_weights()
        for i in range(1, self.transmitters + 1):
            getattr(self, f'target_network{i}').set_weights(weight)
        return 0


