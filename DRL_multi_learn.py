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



class DRLmultiagent(object):
    def __init__(self, state_size, action_size, action_cand, pmax, noise):
        self.TTIs = 1000
        self.simul_rounds = 1

        self.EPSILON = 0.2

        self.EPSILON_DECAY = 1-math.pow(10,-4)
        #self.EPSILON_DECAY = 0.99
        self.EPSILON_MIN = 0.01

        self.learning_rate = 5 * math.pow(10, -3)
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

        self.replay_buffer = deque(maxlen=1000)
        self.update_rate = 100
        self.main_network = self.build_network()
        weight = self.main_network.get_weights()

        for i in range(1, self.transmitters + 1):
            setattr(self, f'target_network{i}', self.build_network())
            getattr(self, f'target_network{i}').set_weights(weight)

        self.loss = []



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
            Q_values = getattr(self, f'target_network{agent + 1}').predict(tf.reshape(state, [1, -1]))
            action_temp = np.argmax(Q_values[0])
            action = self.action_set[int(action_temp)]
            print('GRD agent: ', agent, 'power: ', action)

        return action


    def step(self, state, actions, TTI, max_TTI, channel_gain, agent):

        if TTI >= max_TTI:
            done = True
        else:
            done = False




        array_of_interference = state.numpy()

        reward = np.zeros(self.transmitters)
        temp_reward1 = np.zeros(self.transmitters)
        temp_reward2 = np.zeros(self.transmitters)
        direct_signal = np.zeros(self.transmitters)
        for i in range(self.transmitters):
            action_of_agent = actions[i]
            inter = 0
            for j in range(self.transmitters):
                if i == j:
                    direct_signal[i] = channel_gain[i, j] * action_of_agent
                else:
                    inter += array_of_interference[j, i]
            temp_reward1[i] = math.log2(1 + (direct_signal[i]) / (inter + self.noise))

        for i in range(self.transmitters):
            array_of_interference_without_agent = array_of_interference
            array_of_interference_without_agent[i, :] = 0
            for j in range(self.transmitters):
                if i == j:
                    temp_reward2[i] += 0
                else:

                    rate_with_agent = math.log2(1 + (direct_signal[j]) / (np.sum(array_of_interference[:, j]) + self.noise))
                    rate_without_agent = math.log2(1 + (direct_signal[j]) / (np.sum(array_of_interference_without_agent[:, j]) + self.noise))
                    temp_reward2[i] += (rate_without_agent - rate_with_agent)

        for i in range(self.transmitters):
            reward[i] = temp_reward1[i] - temp_reward2[i]


        for i in range(self.transmitters):
            action_of_agent = actions[i]
            inter = 0
            for j in range(self.transmitters):
                if i == j:
                    direct_signal[i] = channel_gain[i, j] * action_of_agent
                    array_of_interference[i, j] = 0
                else:
                    action_of_interferer = actions[j]
                    gain_temp_interferer = channel_gain[j, i]
                    inter_of_interferer = gain_temp_interferer * action_of_interferer
                    array_of_interference[j, i] = inter_of_interferer

        next_state = array_of_interference#.flatten()
        next_state = tf.convert_to_tensor(next_state.reshape(1, -1), dtype=tf.float32)

        info = {}

        return  next_state, reward, done, info




    def train(self, batch_size):

        # sample a mini batch of transition from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        # compute the Q value using the target network
        for state, action, reward, next_state, done, agent in minibatch:
            if not done:
                target_Q = reward + self.gamma * np.amax(getattr(self, f'target_network{agent+1}').predict(tf.reshape(state, [1, -1]))[0])

            else:
                target_Q = reward
            # compute the Q value using the main network
            Q_values = self.main_network.predict(tf.reshape(state, [1, -1]))

            for i in range(len(self.action_set)):
                if action == self.action_set[i]:
                    action_node_number = i

            Q_values[0][action_node_number] = target_Q

            # train the main network
            result = self.main_network.fit(tf.reshape(state, [1, -1]), Q_values, epochs=1, verbose=1)

            self.loss.append(result.history['loss'])

        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= self.EPSILON_DECAY



    def update_target_network(self):
        weight = self.main_network.get_weights()
        for i in range(1, self.transmitters + 1):
            getattr(self, f'target_network{i}').set_weights(weight)
        return 0


