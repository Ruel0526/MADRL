import tensorflow as tf
from DRL_env import DRLenv
from DRL_learn import DRLagent
from DRL_multi_learn_r5 import DRLmultiagent
from DRL_multi_MIMO_learn import DRLmultiMIMO
from DRL_env_MIMO import DRLenvMIMO
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import math
import matplotlib.pyplot as plt
import cvxpy as cp
from tqdm import tqdm
from scipy.optimize import minimize
import time
import sys
import itertools as it
from itertools import product
import random
import scipy.special as sp
import pandas as pd

import tracemalloc

from memory_profiler import profile

import objgraph
import gc

import matplotlib.patches as patches

'''
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import sys
'''
'''
# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

if use_cuda:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')


# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

'''


def main():
    num_TTIs = 500
    num_simul_rounds = 1

    batch_size = 8
    env = DRLenv()
    dqn = DRLagent(3, 1000)

    done = False
    TTI = 0

    rewards = np.zeros((num_simul_rounds, num_TTIs))

    for i in range(num_simul_rounds):
        Return = 0
        state = np.zeros(3)
        action = np.zeros(3)

        for j in range(num_TTIs):

            if j % dqn.update_rate == 0:
                dqn.update_target_network()

            starter = time.time()
            action = dqn.epsilon_greedy(state)
            end = time.time()
            print('time =', end - starter)
            next_state, reward, done, _ = dqn.step(state, action, j, num_TTIs)

            dqn.store_transistion(state, action, reward, next_state, done)

            state = next_state

            Return += reward
            '''
            if j == 0:
                cumul_reward[i, j] = Return
            else:
                cumul_reward[i, j] = Return/j
            '''
            rewards[i, j] = reward
            print('next_state', next_state, 'action', action)

            print('Iteration:', j, ',' 'Reward', reward)

            if done:
                print('Simul round:', i, ',' 'Return', Return)
                break

            if len(dqn.replay_buffer) > batch_size:
                dqn.train(batch_size)

    reward_avg = rewards.sum(axis=0) / num_simul_rounds

    # np.save('./save_weights/centralized_DRL.npy', rewards)
    # np.save('./save_weights/centralized_DRL_test.npy', rewards)


def main_multi_MIMO():
    num_simul_rounds = 1
    num_TTIs = 10000

    batch_size = 8
    env = DRLenvMIMO()
    dqn_multi = DRLmultiMIMO(19, 190)

    done = False
    TTI = 0

    rewards = np.zeros((num_simul_rounds, num_TTIs))

    f_d = 10
    T = 0.2
    rho = sp.jv(0, 2 * math.pi * f_d * T)
    transmitters = 19
    cell = 19
    antenna = 10

    users = 4
    power_cand = 10
    pmax = 6.30957  # 38dbm
    user_selection_num = 2
    power_set = np.linspace(0, pmax, power_cand)
    user_set = np.arange(0, users, 1)
    action_set_temp = np.arange(0, users * power_cand, 1)
    action_set = list(it.combinations(action_set_temp, user_selection_num))

    for i in range(num_simul_rounds):
        Return = 0

        states_of_agents = np.zeros((transmitters, users))
        actions_of_agents = []
        for j in range(transmitters):
            actions_of_agents.append((0, 0))
        '''
        actions_of_agents_opt = []
        for j in range(transmitters):
            actions_of_agents_opt.append(0)
        actions_of_agents_opt_delay = []
        for j in range(transmitters):
            actions_of_agents_opt_delay.append(0)
        '''
        H = np.ones((transmitters, cell, antenna, users)) * (
                    random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        # prev_H = np.ones((transmitters, cell, antenna, users)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        epsilon = 0.1

        # epsilon_decay = 1-math.pow(10,-4)
        epsilon_decay = 1
        epsilon_min = 0.01
        for j in range(num_TTIs):

            if j % dqn_multi.update_rate == 0:
                dqn_multi.update_target_network()

            for k in range(transmitters):
                actions_of_agents[k] = dqn_multi.epsilon_greedy(states_of_agents[k, :], epsilon)

            for x in range(transmitters):
                for y in range(cell):
                    for z in range(antenna):
                        for w in range(users):
                            innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                            htemp = rho * H[x, y, z, w] + (math.sqrt(1 - math.pow(rho, 2)) * innov)
                            H[x, y, z, w] = htemp

            reward_temp = np.zeros((transmitters))
            for k in range(transmitters):
                next_state, reward, done, info = dqn_multi.step(states_of_agents[k, :], actions_of_agents, j, num_TTIs, k, H)
                dqn_multi.store_transistion(states_of_agents[k, :], actions_of_agents[k], reward, next_state, done)
                states_of_agents[k, :] = next_state
                reward_temp[k] = reward

            final_reward = np.sum(reward_temp)

            for x in range(transmitters):
                for y in range(cell):
                    for z in range(antenna):
                        for w in range(users):
                            innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                            htemp = rho * H[x, y, z, w] + (math.sqrt(1 - math.pow(rho, 2)) * innov)
                            H[x, y, z, w] = htemp

            Return += final_reward
            '''
            if j == 0:
                cumul_reward[i, j] = Return
            else:
                cumul_reward[i, j] = Return/j
            '''
            rewards[i, j] = final_reward
            # print('next_state', next_state, 'action', action)

            print('Iteration:', j, ',' 'Reward', final_reward)

            if done:
                print('Simul round:', i, ',' 'Return', Return)
                break

            if len(dqn_multi.replay_buffer) > batch_size:
                dqn_multi.train(batch_size)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

    # np.save('./save_weights/multi_agent_DRL_MIMO.npy', rewards)
    # np.save('./save_weights/multi_agent_DRL_test.npy', rewards)


'''
def opt_MIMO():
    num_simul_rounds = 1
    num_TTIs = 1000

    batch_size = 8
    env = DRLenvMIMO()
    dqn_multi = DRLmultiMIMO(19, 190)

    done = False
    TTI = 0

    rewards = np.zeros((num_simul_rounds, num_TTIs))

    f_d = 10
    T = 0.2
    rho = sp.jv(0, 2*math.pi*f_d*T)

    self.pmax = 6.30957 #38dbm


    power_cand = 5
    power_set = np.linspace(0, self.pmax, self.power_cand)
    self.user_set = np.arange(0, self.users, 1)
        self.action_set_temp = np.arange(0, self.users * self.power_cand, 1)
        self.action_set = list(it.combinations(self.action_set_temp, self.user_selection_num))

    for i in range(num_simul_rounds):
        Return = 0
        transmitters = 19
        cell = 19
        antenna = 10
        users = 4



        states_of_agents = np.zeros((transmitters, users))
        actions_of_agents = []
        for j in range(transmitters):
            actions_of_agents.append((0,0))



        H = np.ones((transmitters, cell, antenna, users)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        prev_H = np.ones((transmitters, cell, antenna, users)) * (random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)

        for j in range(num_TTIs):

            for x in range(transmitters):
                for y in range(cell):
                    for z in range(antenna):
                        for w in range(users):
                            innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                            htemp = rho * H[x,y,z,w] + (math.sqrt(1-math.pow(rho, 2)) * innov)
                            H[x,y,z,w] = htemp

            optimal[i, j] = 0
            for k in range(transmitters):
                action_of_agent = actions[k]
                powers_of_agent = np.zeros((user_selection_num))
                user_index_of_agent = np.zeros((user_selection_num))

        for i in range(self.user_selection_num):
            user_index_of_agent[i] = action_of_agent[i] % self.users
            powers_of_agent[i] = self.power_set[int(action_of_agent[i] // self.users)]

        selected_H = self.scheduled_csi(user_index_of_agent, H[agent, agent, :, :])
        direct_signal = np.zeros((self.user_selection_num))
        for i in range(self.user_selection_num):
            gain_temp = self.env.channel_gain(self.A[agent], self.B[agent][int(user_index_of_agent[i])], selected_H[:,i])
            F_bb = self.digital_precoder(selected_H[:,i])
            direct_signal[i] = gain_temp @ F_bb * powers_of_agent[i]



        inter = np.zeros((self.users))
        for i in range(self.users):
            inter_temp_temp = 0
            for j in range(self.transmitters):
                if j == agent:
                    inter_temp_temp += 0
                else:
                    action_of_interferer = actions[j]
                    user_index_of_interferer = np.zeros((self.user_selection_num))
                    power_of_interferer = np.zeros((self.user_selection_num))
                    for k in range(self.user_selection_num):
                        user_index_of_interferer[k] = action_of_interferer[k] % self.users
                        power_of_interferer[k] = self.power_set[int(action_of_interferer[k] // self.users)]
                    selected_H_interferer = self.scheduled_csi(user_index_of_interferer, H[j, j, :, :])
                    Fbb_interferer = np.zeros((self.user_selection_num, self.antenna))
                    for k in range(self.user_selection_num):
                        Fbb_interferer[k, :] = np.array(self.digital_precoder(selected_H_interferer[:,k])).flatten()

                    for k in range(self.user_selection_num):
                        gain_temp_interferer = self.env.channel_gain(self.A[j], self.B[agent][k], H[j, agent, :, i])
                        inter_of_interferer = gain_temp_interferer @ Fbb_interferer[k,:] * power_of_interferer[k]
                        inter_temp_temp += inter_of_interferer

            inter[i] = inter_temp_temp


        next_state = inter


        sum_rate = 0
        reward = 0

        for i in range(self.user_selection_num):
            SINR_temp = (np.abs(direct_signal[i]))/(np.abs(state[int(user_index_of_agent[i])]) + self.noise)
            reward += math.log(1+SINR_temp)

            best1 = 0
            best2 = 0
            best3 = 0



            final_reward = np.sum(reward_temp)

            Return += final_reward

            rewards[i, j] = final_reward
            #print('next_state', next_state, 'action', action)

            print('Iteration:', j, ',' 'Reward', final_reward)

            if done:
                print('Simul round:', i, ',' 'Return', Return)
                break

            if len(dqn_multi.replay_buffer) > batch_size:
                dqn_multi.train(batch_size)

    reward_avg = rewards.sum(axis=0) / num_simul_rounds

    np.save('./save_weights/multi_agent_DRL_MIMO.npy', rewards)
    #np.save('./save_weights/multi_agent_DRL_test.npy', rewards)


'''
def compute_sum_rate(channel_gain, actions, noise):
    sum_rate = 0
    SINR_cap = 10 ** (30 / 10)
    for i in range(len(actions)):
        interferences = sum(channel_gain[j, i] * actions[j] for j in range(len(actions)) if j != i)
        SINR = channel_gain[i, i] * actions[i] / (interferences + noise)

        # Cap the SINR at 30 dB
        SINR = min(SINR, SINR_cap)
        #if SINR == SINR_cap:
            #print('It is over 30dB')

        sum_rate += math.log(1 + SINR)
    return sum_rate


def find_optimal_actions(channel_gain, action_set, noise ,num_agents):

    optimal_sum_rate = float('-inf')
    best_actions = [0] * num_agents

    # Iterate over all combinations of actions for each agent
    for actions in it.product(action_set, repeat=num_agents):
        current_sum_rate = compute_sum_rate(channel_gain, actions, noise)
        if current_sum_rate > optimal_sum_rate:
            optimal_sum_rate = current_sum_rate
            best_actions = actions

    return best_actions, optimal_sum_rate


def sum_rate_objective(power, channel_gain_matrix, noise_power):
    num_links = channel_gain_matrix.shape[0]
    total_rate = 0
    SINR_cap = 10 ** (30 / 10)  # Convert 30 dB SINR cap to linear scale
    for i in range(num_links):
        signal = channel_gain_matrix[i, i] * power[i]
        interference = noise_power + sum(channel_gain_matrix[j, i] * power[j] for j in range(num_links) if j != i)
        SINR = signal / interference
        SINR = min(SINR, SINR_cap)
        rate = np.log2(1 + SINR)
        total_rate += rate
    return -total_rate  # Minimize the negative rate for maximization

def maximize_sum_rate_FP(num_links, channel_gain_matrix, noise_power, pmax):
    initial_power = np.full(num_links, pmax / num_links)  # Initial guess for power allocation

    # Define bounds for each power variable: non-negative and not exceeding pmax
    bounds = [(0, pmax) for _ in range(num_links)]

    # Minimize the negative sum rate
    result = minimize(sum_rate_objective, initial_power, args=(channel_gain_matrix, noise_power),
                      bounds=bounds, method='SLSQP', options={'disp': True})

    # Power allocation and achieved rates
    optimal_power = result.x #if result.success else None
    achieved_rates = -sum_rate_objective(optimal_power, channel_gain_matrix, noise_power) #if result.success else None

    return optimal_power, achieved_rates





def main_multi():

    num_simul_rounds = 1
    num_TTIs = 3000

    batch_size = 256
    env = DRLenv()

    done = False

    f_d = 10
    T = 0.02
    rho = sp.jv(0, 2 * math.pi * f_d * T)
    transmitters = 19
    users = 19
    pmax = math.pow(10, 0.8)  # 38dbm
    action_cand = 10
    action_set = np.linspace(0, pmax, action_cand)
    noise = math.pow(10, -14.4)

    interferer_size = 5

    state_number = 7+4*interferer_size+3*interferer_size

    dqn_multi = DRLmultiagent(state_number, 10, action_cand, pmax, noise)

    rewards = np.zeros((num_simul_rounds, num_TTIs))
    sum_rate_of_DRL = np.zeros((num_simul_rounds, num_TTIs))
    optimal = np.zeros((num_simul_rounds, num_TTIs))
    optimal_no_delay = np.zeros((num_simul_rounds, num_TTIs))
    full_pwr = np.zeros((num_simul_rounds, num_TTIs))
    random_pwr = np.zeros((num_simul_rounds, num_TTIs))

    action_full_pwr = np.ones((transmitters)) * pmax

    FP = np.zeros((num_simul_rounds, num_TTIs))
    central = np.zeros((num_simul_rounds, num_TTIs))

    for i in range(num_simul_rounds):
        Return = 0
        states_of_agents = np.zeros((transmitters, state_number))  # .flatten()
        # states_of_agents = tf.convert_to_tensor(states_of_agents.reshape(1, -1), dtype=tf.float32)



        actions_of_agents = np.zeros((transmitters))

        H = np.ones((transmitters, transmitters)) * (
                    random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        channel_gain = np.zeros((transmitters, users))
        for x in range(transmitters):
            for y in range(users):
                channel_gain[x, y] = env.channel_gain(dqn_multi.A[x], dqn_multi.B[y], H[x, y])

        for x in range(transmitters):
            states_of_agents[x, 0] = x
            states_of_agents[x, 3] = channel_gain[x, x]

        epsilon_min = 0.01
        lambda_epsilon = 1e-4
        epsilon = 0.2  # Initial epsilon

        best = np.zeros((transmitters))

        optimal_power = np.zeros((transmitters))

        action_random = np.zeros((transmitters))

        state_transit = np.zeros((transmitters, state_number))

        for j in range(num_TTIs):

            print(epsilon)


            #optimal[i, j] = compute_sum_rate(channel_gain, best, noise)

            #central[i, j] = compute_sum_rate(channel_gain, optimal_power, noise)



            #best, optimal_no_delay[i, j] = find_optimal_actions(channel_gain, action_set, noise, transmitters)

            #print('best actions of OPT = ', best)



            for k in range(transmitters):
                actions_of_agents[k] = dqn_multi.epsilon_greedy(states_of_agents[k, :], epsilon)

            print('DRL actions:', actions_of_agents)
            '''
            optimal_power, rates = maximize_sum_rate_FP(transmitters, channel_gain, noise, pmax)
            print("Optimal power FP:", optimal_power)
            FP[i, j] = rates
            '''
            #print("Achieved Rates:", rates)

            full_pwr[i, j] = compute_sum_rate(channel_gain, action_full_pwr, noise)

            action_random.fill(0)
            for x in range(transmitters):
                action_random[x] = action_set[random.randint(0, action_cand - 1)]

            random_pwr[i, j] = compute_sum_rate(channel_gain, action_random, noise)

            old_channel_gain = np.copy(channel_gain)

            for x in range(transmitters):
                for y in range(users):
                    innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                    htemp = rho * H[x, y] + (math.sqrt(1 - math.pow(rho, 2)) * innov)
                    H[x, y] = htemp


            for x in range(transmitters):
                for y in range(users):
                    channel_gain[x, y] = env.channel_gain(dqn_multi.A[x], dqn_multi.B[y], H[x, y])
                    #print("Tx poisition of ", x, dqn_multi.A[x])
                    #print("Rx poisition of ", y, dqn_multi.B[y])

            final_reward = 0
            #tracemalloc.start()
            for k in range(transmitters):
                # print('iteration =', j, 'agent=', k, 'current state =', states_of_agents[k, :])
                # print('iteration =', j, 'agent=', k, 'new state=', next_state)

                next_state, reward, done, info = dqn_multi.step(states_of_agents[k, :], actions_of_agents, j, num_TTIs,
                                                                old_channel_gain, channel_gain, k)
                #print('iter', j, 'agent', k, 'reward', reward)
                #objgraph.show_growth()
                #print('iteration =', j, 'agent=', k, 'current state =', states_of_agents[k, :])
                #print('iteration =', j, 'agent=', k, 'new state=', next_state)
                dqn_multi.store_transition(states_of_agents[k, :], actions_of_agents[k], reward, next_state, done)
                #print('iteration', j, 'agent', k, 'current', states_of_agents[k, :])
                #print('iteration', j, 'agent', k, 'next', next_state)
                #objgraph.show_growth()
                state_transit[k, :] = np.copy(next_state)
                final_reward += reward
                sum_rate_of_DRL[i, j] += dqn_multi.temp_reward1
            #snapshot = tracemalloc.take_snapshot()
            #top_stats = snapshot.statistics('lineno')

            #print("[ Top 10 ]")
            #for stat in top_stats[:10]:
            #    print(stat)

            states_of_agents = state_transit

            del old_channel_gain
            del next_state

            Return += final_reward
            rewards[i, j] = final_reward

            print('Iteration:', j, ',' 'Reward', rewards[i, j])
            print('Iteration:', j, ',' 'Sum rate of DRL', sum_rate_of_DRL[i, j])
            #print('Iteration:', j, ',' 'OPT Reward', optimal[i, j])
            #print('Iteration:', j, ',' 'OPT (no delay) Reward', optimal_no_delay[i, j])
            print('Iteration:', j, ',' 'Full Power Reward', full_pwr[i, j])
            print('Iteration:', j, ',' 'Random Power Reward', random_pwr[i, j])
            #print('Iteration:', j, ',' 'FP (delay)', central[i, j])
            #print('Iteration:', j, ',' 'FP (no delay)', FP[i, j])
            if done:  # 같은 TTI의 step func에서도 done은 세번 갱신된다.
                print('Simul round:', i, ',' 'Return', Return)
                break

            if len(dqn_multi.replay_buffer) > batch_size:
                print("Training is triggered.")
                dqn_multi.train(batch_size)

            if j % dqn_multi.update_rate == 0:
                tau = 0.001  # You can adjust this value
                dqn_multi.soft_update_target_network(tau)

            epsilon = max(epsilon_min, (1 - lambda_epsilon) * epsilon)

            gc.collect()

            print(f"Time Slot {j}: Replay Buffer Length = {len(dqn_multi.replay_buffer)}")
            #print(dqn_multi.learning_rate)

    #np.save('./save_weights/FP.npy', FP)
    #np.save('./save_weights/central.npy', central)
    np.save('./save_weights/full_power.npy', full_pwr)
    np.save('./save_weights/random_power.npy', random_pwr)
    np.save('./save_weights/multi_agent_DRL.npy', rewards)
    np.save('./save_weights/multi_agent_DRL_rate.npy', sum_rate_of_DRL)
    np.save('./save_weights/optimal_no_delay.npy', optimal_no_delay)
    np.save('./save_weights/optimal.npy', optimal)
    # np.save('./save_weights/multi_agent_DRL_test.npy', rewards)

    plt.plot(dqn_multi.loss)
    plt.show()


'''
def objective(x):
    return -math.log(1+(H[0,0] * x[0])/(H[1,0] * x[1] + H[2,0] * x[2] + noise)) - math.log(1+(H[1,1] * x[2])/(H[0,1] * x[0] + H[2,1] * x[2] + noise)) - math.log(1+ (H[2,2] * x[2])/(H[0,2] * x[0] + H[1,2] * x[1] + noise))



def fractional2():
    num_TTIs = 2000
    num_simul_rounds = 1

    dqn = DRLagent(3, 64)
    noise = math.pow(10,-11.4)
    pmax = 6.30957

    optimal = np.zeros((num_simul_rounds, num_TTIs))

    for i in range(num_simul_rounds):
        for j in range(num_TTIs):
            H = np.zeros((3, 3))
            env = DRLenv(j+1)

            A = env.tx_positions_gen()
            B = env.rx_positions_gen(A)

            for l in range(3):
                for m in range(3):
                    temp = env.Jakes_channel(A[l], B[m])
                    H[l, m] = temp

            starting_point = [0, 0, 0]
            x =

            result = minimize(objective, starting_point)

            solution = result['x']
            evaluation = -1 * objective(solution)

            optimal[i,j] = evaluation

    print(evaluation)
'''




def full_pwr():
    num_simul_rounds = 1
    num_TTIs = 1000

    env = DRLenv()

    f_d = 10
    T = 0.02
    rho = sp.jv(0, 2 * math.pi * f_d * T)
    transmitters = 19
    users = 19
    pmax = math.pow(10, 0.8)  # 38dbm
    action_cand = 4
    noise = math.pow(10, -14.4)

    interferer_size = 2

    state_number = 7 + 4 * interferer_size + 3 * interferer_size

    dqn_multi = DRLmultiagent(state_number, 10, action_cand, pmax, noise)

    full_pwr = np.zeros((num_simul_rounds, num_TTIs))
    optimal_no_delay = np.zeros((num_simul_rounds, num_TTIs))

    action_full_pwr = np.ones((transmitters)) * pmax

    action_set = np.linspace(0, pmax, action_cand)


    for i in range(num_simul_rounds):


        H = np.ones((transmitters, transmitters)) * (
                random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        channel_gain = np.zeros((transmitters, users))
        for x in range(transmitters):
            for y in range(users):
                channel_gain[x, y] = env.channel_gain(dqn_multi.A[x], dqn_multi.B[y], H[x, y])

        for x in range(transmitters):
            x1, y1 = dqn_multi.A[x]
            x2, y2 = dqn_multi.B[x]
            print(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))



        #print(dqn_multi.A)
        #print(dqn_multi.B)
        #print(H)
        #print(channel_gain)


        for j in range(num_TTIs):

            #best, optimal_no_delay[i, j] = find_optimal_actions(channel_gain, action_set, noise, transmitters)

            #print('best actions of OPT = ', best)

            full_pwr[i, j] = compute_sum_rate(channel_gain, action_full_pwr, noise)
            #optimal_power, rates = maximize_sum_rate_FP(transmitters, channel_gain, noise, pmax)
            #print("Optimal power FP:", optimal_power)
            #FP[i, j] = rates


            for x in range(transmitters):
                for y in range(users):
                    innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                    htemp = rho * H[x, y] + (math.sqrt(1 - math.pow(rho, 2)) * innov)
                    H[x, y] = htemp

            for x in range(transmitters):
                for y in range(users):
                    channel_gain[x, y] = env.channel_gain(dqn_multi.A[x], dqn_multi.B[y], H[x, y])

            #print(H)
            #print(channel_gain)

            print('Iteration:', j, ',' 'Full Power Reward', full_pwr[i, j]/transmitters)
            #print('Iteration:', j, ',' 'optimal Reward', optimal_no_delay[i, j] / transmitters)

        print('Average', np.sum(full_pwr[i, :]/(num_TTIs*transmitters)))
        #print('Average FP', np.sum(optimal_no_delay[i, :] / (num_TTIs * transmitters)))

    # np.save('./save_weights/FP.npy', optimal)
    np.save('./save_weights/full_power.npy', full_pwr)


def moving_average(rewards, window_size):
    cumsum = np.cumsum(np.insert(rewards, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def graph(switch):
    centralized_DRL = np.load('./save_weights/centralized_DRL_test.npy')
    multi_agent_DRL = np.load('./save_weights/multi_agent_DRL.npy')
    multi_agent_DRL_MIMO = np.load('./save_weights/multi_agent_DRL_MIMO.npy')
    FP = np.load('./save_weights/FP.npy')
    delayed_FP = np.load('./save_weights/central.npy')
    optimal = np.load('./save_weights/optimal.npy')
    optimal_no_delay = np.load('./save_weights/optimal_no_delay.npy')
    full_pwr = np.load('./save_weights/full_power.npy')
    random_pwr = np.load('./save_weights/random_power.npy')
    rate_DRL = np.load('./save_weights/multi_agent_DRL_rate.npy')

    num_tx = 19
    num_simul_rounds = 1
    start = 20
    space = 250

    reward_avg = centralized_DRL.sum(axis=0) / (num_simul_rounds * num_tx)
    reward_avg_multi = multi_agent_DRL.sum(axis=0) / (num_simul_rounds * num_tx)
    reward_avg_multi_rate = rate_DRL.sum(axis=0) / (num_simul_rounds * num_tx)
    reward_avg_multi_MIMO = multi_agent_DRL_MIMO.sum(axis=0) / (num_simul_rounds * num_tx)
    reward_avg_FP = FP.sum(axis=0) / (num_simul_rounds * num_tx)
    reward_avg_delayed_FP = delayed_FP.sum(axis=0) / (num_simul_rounds * num_tx)
    reward_avg_optimal = optimal.sum(axis=0) / (num_simul_rounds * num_tx)
    reward_avg_optimal_no_delay = optimal_no_delay.sum(axis=0) / (num_simul_rounds * num_tx)
    reward_avg_full_pwr = full_pwr.sum(axis=0) / (num_simul_rounds * num_tx)
    reward_avg_random_pwr = random_pwr.sum(axis=0) / (num_simul_rounds * num_tx)
    '''
    for i in range(len(reward_avg)):
        if reward_avg[i] ==0 and i != 0:
            reward_avg[i] = reward_avg[i-1]

    for i in range(len(reward_avg_multi)):
        if reward_avg_multi[i] ==0 and i != 0:
            reward_avg_multi[i] = reward_avg_multi[i-1]
    '''
    '''
    cumulative_rewards = [np.mean(reward_avg[:i + 1]) for i in range(start, len(reward_avg))]
    cumulative_rewards_multi = [np.mean(reward_avg_multi[:i + 1]) for i in range(start, len(reward_avg_multi))]
    #cumulative_rewards_multi = [np.mean((reward_avg_multi[i - 100:i + 1])) for i in range(100, len(reward_avg_multi))]
    cumulative_rewards_FP = [np.mean(reward_avg_FP[:i + 1]) for i in range(start, len(reward_avg_FP))]
    cumulative_rewards_multi_MIMO = [np.mean(reward_avg_multi_MIMO[:i + 1]) for i in range(start, len(reward_avg_multi_MIMO))]

    cumulative_rewards_optimal = [np.mean(reward_avg_optimal[:i + 1]) for i in range(start, len(reward_avg_optimal))]
    cumulative_rewards_optimal_no_delay = [np.mean(reward_avg_optimal_no_delay[:i + 1]) for i in range(start, len(reward_avg_optimal_no_delay))]
    cumulative_rewards_full_pwr = [np.mean(reward_avg_full_pwr[:i + 1]) for i in
                                   range(start, len(reward_avg_full_pwr))]
    cumulative_rewards_random_pwr = [np.mean(reward_avg_random_pwr[:i + 1]) for i in
                                   range(start, len(reward_avg_full_pwr))]

    cumulative_rate_multi = [np.mean(reward_avg_multi_rate[:i + 1]) for i in range(start, len(reward_avg_multi_rate))]

    '''
    '''
    cumulative_rate_multi = [np.mean((reward_avg_multi_rate[i - space:i + 1])) for i in range(space, len(reward_avg_multi_rate))]
    cumulative_rewards_optimal = [np.mean((reward_avg_optimal[i - space:i + 1])) for i in range(space, len(reward_avg_optimal))]
    cumulative_rewards_optimal_no_delay = [np.mean((reward_avg_optimal_no_delay[i - space:i + 1])) for i in
                                  range(space, len(reward_avg_optimal_no_delay))]
    cumulative_rewards_full_pwr = [np.mean((reward_avg_full_pwr[i - space:i + 1])) for i in
                                  range(space, len(reward_avg_full_pwr))]
    cumulative_rewards_random_pwr = [np.mean((reward_avg_random_pwr[i - space:i + 1])) for i in
                                   range(space, len(reward_avg_random_pwr))]
    '''

    if switch == 0:
        cumulative_rewards = [np.mean(reward_avg[:i + 1]) for i in range(start, len(reward_avg))]
        cumulative_rewards_multi = [np.mean(reward_avg_multi[:i + 1]) for i in range(start, len(reward_avg_multi))]

        cumulative_rewards_multi_MIMO = [np.mean(reward_avg_multi_MIMO[:i + 1]) for i in
                                         range(start, len(reward_avg_multi_MIMO))]

        cumulative_rewards_optimal = [np.mean(reward_avg_optimal[:i + 1]) for i in
                                      range(start, len(reward_avg_optimal))]
        cumulative_rewards_optimal_no_delay = [np.mean(reward_avg_optimal_no_delay[:i + 1]) for i in
                                               range(start, len(reward_avg_optimal_no_delay))]
        cumulative_rewards_full_pwr = [np.mean(reward_avg_full_pwr[:i + 1]) for i in
                                       range(start, len(reward_avg_full_pwr))]
        cumulative_rewards_random_pwr = [np.mean(reward_avg_random_pwr[:i + 1]) for i in
                                         range(start, len(reward_avg_full_pwr))]

        cumulative_rate_multi = [np.mean(reward_avg_multi_rate[:i + 1]) for i in
                                 range(start, len(reward_avg_multi_rate))]
        cumulative_rewards_FP = [np.mean(reward_avg_FP[:i + 1]) for i in range(start, len(reward_avg_FP))]
        cumulative_rewards_delayed_FP = [np.mean(reward_avg_delayed_FP[:i + 1]) for i in range(start, len(reward_avg_delayed_FP))]

    if switch == 1:
        cumulative_rewards_multi = moving_average(reward_avg_multi, space)
        cumulative_rate_multi = moving_average(reward_avg_multi_rate, space)
        cumulative_rewards_optimal = moving_average(reward_avg_optimal, space)
        cumulative_rewards_optimal_no_delay = moving_average(reward_avg_optimal_no_delay, space)
        cumulative_rewards_full_pwr = moving_average(reward_avg_full_pwr, space)
        cumulative_rewards_random_pwr = moving_average(reward_avg_random_pwr, space)
        cumulative_rewards_FP = moving_average(reward_avg_FP, space)
        cumulative_rewards_delayed_FP = moving_average(reward_avg_delayed_FP, space)

    plt.subplot(1, 1, 1)

    # plt.plot(range(start, len(reward_avg)), cumulative_rewards, label='Centralized DRL')

    # plt.plot(range(10, len(reward_avg_multi_rate)), cumulative_rate_multi, label='Multi-agent DRL')
    # plt.plot(range(start, len(reward_avg_multi_MIMO)), cumulative_rewards_multi_MIMO, label='Multi-agent DRL MIMO')
    # plt.plot(range(100, len(reward_avg_multi)), cumulative_rewards_multi, label='Multi-agent DRL')
    '''
    plt.plot(range(start, len(reward_avg_multi)), cumulative_rewards_multi, label='Multi-agent DRL')
    #plt.plot(range(start, len(reward_avg_multi_rate)), cumulative_rate_multi, label='Multi-agent DRL')
    plt.plot(range(start, len(reward_avg_optimal_no_delay)), cumulative_rewards_optimal_no_delay, label='OPT (no delay)')
    plt.plot(range(start, len(reward_avg_optimal)), cumulative_rewards_optimal, label='OPT (delay)')
    plt.plot(range(start, len(reward_avg_full_pwr)), cumulative_rewards_full_pwr, label='full power')
    plt.plot(range(start, len(reward_avg_random_pwr)), cumulative_rewards_random_pwr, label='random power')
    '''
    # plt.plot(range(space, len(reward_avg_multi_rate)), cumulative_rate_multi, label='Multi-agent DRL')
    # plt.plot(range(space, len(reward_avg_optimal_no_delay)), cumulative_rewards_optimal_no_delay, label='Brute (no delay)')
    # plt.plot(range(space, len(reward_avg_optimal)), cumulative_rewards_optimal, label='Brute (delay)')
    # plt.plot(range(space, len(reward_avg_full_pwr)), cumulative_rewards_full_pwr, label='full power')
    # plt.plot(range(space, len(reward_avg_random_pwr)), cumulative_rewards_random_pwr, label='random power')

    if switch == 0:
        # plt.plot(range(start, len(reward_avg_multi)), cumulative_rewards_multi, label='Multi-agent DRL')
        plt.plot(range(start, len(reward_avg_multi)), cumulative_rate_multi, label='Multi-agent DRL')
        #plt.plot(range(start, len(reward_avg_optimal_no_delay)), cumulative_rewards_optimal_no_delay,
        #         label='Brute (no delay)')
        #plt.plot(range(start, len(reward_avg_optimal)), cumulative_rewards_optimal, label='Brute (delay)')
        plt.plot(range(start, len(reward_avg_full_pwr)), cumulative_rewards_full_pwr, label='Full power')
        plt.plot(range(start, len(reward_avg_random_pwr)), cumulative_rewards_random_pwr, label='Random power')
        #plt.plot(range(start, len(reward_avg_FP)), cumulative_rewards_FP, label='FP (no delay)')
        #plt.plot(range(start, len(reward_avg_delayed_FP)), cumulative_rewards_delayed_FP, label='Central (Delayed FP)')

    if switch == 1:
        # plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_multi)), cumulative_rewards_multi, label='Multi-agent DRL')
        plt.plot(range(space - 1, space - 1 + len(cumulative_rate_multi)), cumulative_rate_multi,
                 label='Multi-agent DRL')
        #plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_optimal)), cumulative_rewards_optimal,
        #         label='Brute (delay)')
        #plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_optimal_no_delay)),
        #         cumulative_rewards_optimal_no_delay, label='Brute (no delay)')
        plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_full_pwr)), cumulative_rewards_full_pwr,
                 label='Full power')
        plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_random_pwr)), cumulative_rewards_random_pwr,
                 label='Random power')
        #plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_FP)), cumulative_rewards_FP,
        #         label='FP (no delay)')
        #plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_delayed_FP)), cumulative_rewards_delayed_FP,
        #         label='Central (Delayed FP)')

    plt.legend()
    plt.xlabel('Time slot')
    plt.ylabel('Moving average of SE per link')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.show()


def bitcheck():
    is_64bit = sys.maxsize > 2 ** 32
    print(is_64bit)


def calculate_reward(agent, actions, channel_gain, noise, transmitters):
    direct_signal = channel_gain[agent, agent] * actions[agent]
    inter = np.sum(channel_gain[:, agent] * actions) - direct_signal
    temp_reward1 = math.log2(1 + direct_signal / (inter + noise))
    reward = temp_reward1

    for m in range(transmitters):
        if m != agent:
            inter_of_interfered = np.sum(channel_gain[:, m] * actions) - channel_gain[m, m] * actions[m]
            rate_with_agent = math.log2(
                1 + channel_gain[m, m] * actions[m] / (inter_of_interfered + noise))
            inter_of_interfered_without_agent = inter_of_interfered - channel_gain[agent, m] * actions[agent]
            rate_without_agent = math.log2(
                1 + channel_gain[m, m] * actions[m] / (inter_of_interfered_without_agent + noise))
            reward -= (rate_without_agent - rate_with_agent)

    return reward









def plot_hexagonal_grid(tx_positions, rx_positions, inside_status, R, r):
    fig, ax = plt.subplots()
    env = DRLenv()
    # Plot Tx positions and draw hexagons and circles
    for tx in tx_positions:
        ax.scatter(*tx, color='blue', label='Tx' if tx == tx_positions[0] else "")
        # Draw hexagon around each Tx
        hex_vertices = env.hexagon_vertices(tx, R)
        hexagon = patches.Polygon(hex_vertices, fill=False, edgecolor='gray', linestyle='--')
        ax.add_patch(hexagon)
        # Draw circle of radius r around each Tx
        circle = patches.Circle(tx, r, fill=False, edgecolor='green', linestyle=':')
        ax.add_patch(circle)

    # Plot Rx positions
    for rx, inside in zip(rx_positions, inside_status):
        color = 'red' if inside else 'orange'  # Orange for Rx outside the hexagon
        ax.scatter(*rx, color=color, label='Rx (inside)' if inside else 'Rx (outside)')

    ax.set_aspect('equal', 'box')
    ax.legend()
    plt.show()

def testing():
    transmitters = 19
    users = 19
    pmax = math.pow(10, 0.8)  # 38dbm
    action_cand = 10
    noise = math.pow(10, -14.4)

    interferer_size = 2

    state_number = 7 + 4 * interferer_size + 3 * interferer_size

    dqn_multi = DRLmultiagent(state_number, 10, action_cand, pmax, noise)

    A = dqn_multi.A
    B = dqn_multi.B

    for x in range(transmitters):
        x1, y1 = dqn_multi.A[x]
        x2, y2 = dqn_multi.B[x]
        print(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    print(dqn_multi.inside_status)

    plot_hexagonal_grid(A, B, dqn_multi.inside_status, 100, 10)




if __name__ == "__main__":  ##인터프리터에서 실행할 때만 위 함수(main())을 실행해라. 즉, 다른데서 이 파일을 참조할 땐(import 시) def만 가져가고, 실행은 하지말라는 의미.
    # bitcheck()
    # main()
    #main_multi()
    # main_multi_MIMO()
    # opt()
    # fractional()
    full_pwr()


    #graph(0)
    testing()
