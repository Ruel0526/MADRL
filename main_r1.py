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
import gym
from gym.envs.registration import register
import tensorflow as tf
from tf_agents.environments import suite_gym, TFPyEnvironment
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import epsilon_greedy_policy as eg_policy

from DRL_env_r2 import DRLenv2
from DRL_multi_learn_r6 import DRLmultienv

from tensorflow.keras import layers, regularizers

from sklearn.preprocessing import MinMaxScaler

import ray
import ray.rllib
from ray import tune
from ray.rllib.algorithms.dqn import DQN as DQNTrainer
from ray.rllib.algorithms.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
from ray.tune.registry import register_env
from ray.rllib.algorithms.dqn.dqn_tf_policy import build_q_model
#from ray.rllib.execution import ParallelRollouts, ConcatBatches, TrainOneStep, UpdateTargetNetwork

tf1, tf, tfv = try_import_tf()


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
    num_simul_rounds = 1
    num_TTIs = 500

    batch_size = 16
    env = DRLenv()

    done = False

    f_d = 10
    T = 0.02
    rho = sp.jv(0, 2 * math.pi * f_d * T)
    transmitters = 3
    users = 3
    pmax = math.pow(10, 0.8)  # 38dbm
    action_cand = 10
    action_set = np.linspace(0, pmax, action_cand)
    noise = math.pow(10, -14.4)

    state_number = 2*(1+transmitters) + transmitters*users * 2

    evaluation_interval = 20



    rewards = np.zeros((num_simul_rounds, num_TTIs))
    optimal = np.zeros((num_simul_rounds, num_TTIs))
    optimal_no_delay = np.zeros((num_simul_rounds, num_TTIs))
    full_pwr = np.zeros((num_simul_rounds, num_TTIs))
    random_pwr = np.zeros((num_simul_rounds, num_TTIs))

    action_full_pwr = np.ones((transmitters)) * pmax

    FP = np.zeros((num_simul_rounds, num_TTIs))
    central = np.zeros((num_simul_rounds, num_TTIs))

    eval_loss_list = []
    train_loss_list = []

    for i in range(num_simul_rounds):
        dqn = DRLagent(state_number, action_cand**transmitters, action_cand, pmax, noise)
        Return = 0
        states_of_agents = np.zeros((state_number))  # .flatten()
        # states_of_agents = tf.convert_to_tensor(states_of_agents.reshape(1, -1), dtype=tf.float32)

        for x in range(transmitters):
            actions_of_agents = action_set[random.randint(0, action_cand - 1)]




        H = np.ones((transmitters, transmitters)) * (
                    random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        channel_gain = np.zeros((transmitters, users))
        for x in range(transmitters):
            for y in range(users):
                channel_gain[x, y] = env.channel_gain(dqn.A[x], dqn.B[y], H[x, y])

        epsilon_min = 0.01
        lambda_epsilon = 1e-4
        epsilon = 0.4  # Initial epsilon

        best = np.zeros((transmitters))

        action_random = np.zeros((transmitters))

        for j in range(num_TTIs):

            print(epsilon)

            optimal[i, j] = compute_sum_rate(channel_gain, best, noise)

            # central[i, j] = compute_sum_rate(channel_gain, optimal_power, noise)

            best, optimal_no_delay[i, j] = find_optimal_actions(channel_gain, action_set, noise, transmitters)

            print('best actions of OPT = ', best)

            actions_of_agents = dqn.epsilon_greedy(states_of_agents, epsilon)

            print('DRL actions:', actions_of_agents)
            '''
            optimal_power, rates = maximize_sum_rate_FP(transmitters, channel_gain, noise, pmax)
            print("Optimal power FP:", optimal_power)
            FP[i, j] = rates
            '''
            # print("Achieved Rates:", rates)

            full_pwr[i, j] = compute_sum_rate(channel_gain, action_full_pwr, noise)

            action_random.fill(0)
            for x in range(transmitters):
                action_random[x] = action_set[random.randint(0, action_cand - 1)]

            random_pwr[i, j] = compute_sum_rate(channel_gain, action_random, noise)

            old_channel_gain = np.copy(channel_gain)
            '''
            for x in range(transmitters):
                for y in range(users):
                    innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                    htemp = rho * H[x, y] + (math.sqrt(1 - math.pow(rho, 2)) * innov)
                    H[x, y] = htemp

            for x in range(transmitters):
                for y in range(users):
                    channel_gain[x, y] = env.channel_gain(dqn.A[x], dqn.B[y], H[x, y])
                    # print("Tx poisition of ", x, dqn_multi.A[x])
                    # print("Rx poisition of ", y, dqn_multi.B[y])
            '''

            #print('current state =', states_of_agents)

            next_state, reward, done, info = dqn.step(states_of_agents, actions_of_agents, j, num_TTIs,
                                                      old_channel_gain, channel_gain)

            #print('current state =', states_of_agents)

            #print('new state = ', next_state)
            dqn.store_transition(states_of_agents, actions_of_agents, reward, next_state, done)







            Return += reward
            rewards[i, j] = reward

            print('Simul', i, 'Iteration:', j, ',' 'Reward', rewards[i, j])
            print('Simul', i, 'Iteration:', j, ',' 'OPT Reward', optimal[i, j])
            print('Simul', i, 'Iteration:', j, ',' 'OPT (no delay) Reward', optimal_no_delay[i, j])
            print('Simul', i, 'Iteration:', j, ',' 'Full Power Reward', full_pwr[i, j])
            print('Simul', i, 'Iteration:', j, ',' 'Random Power Reward', random_pwr[i, j])
            # print('Iteration:', j, ',' 'FP (delay)', central[i, j])
            # print('Iteration:', j, ',' 'FP (no delay)', FP[i, j])
            if done:  # 같은 TTI의 step func에서도 done은 세번 갱신된다.
                print('Simul round:', i, ',' 'Return', Return)
                break

            if len(dqn.replay_buffer) > batch_size:
                print("Training is triggered.")
                train_loss = dqn.train(batch_size)
                train_loss_list.append(train_loss)

            eval_states = states_of_agents
            eval_actions = actions_of_agents
            eval_rewards = reward
            eval_next_states = next_state
            eval_dones = done

            if j % evaluation_interval == 0:
                eval_minibatch = random.sample(dqn.replay_buffer, min(len(dqn.replay_buffer), batch_size))
                eval_states, eval_actions, eval_rewards, eval_next_states, eval_dones = map(np.array,
                                                                                            zip(*eval_minibatch))
                eval_loss = dqn.evaluate(eval_states, eval_actions, eval_rewards, eval_next_states, eval_dones)
                eval_loss_list.append(eval_loss)
                print("eval loss = ", eval_loss)

            states_of_agents = next_state

            if j % dqn.update_rate == 0:
                tau = 0.001  # You can adjust this value
                dqn.update_target_network()

            epsilon = max(epsilon_min, (1 - lambda_epsilon) * epsilon)

            del old_channel_gain

            gc.collect()

            print(f"Time Slot {j}: Replay Buffer Length = {len(dqn.replay_buffer)}")
            # print(dqn_multi.learning_rate)

    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(eval_loss_list, label='Testing Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # np.save('./save_weights/FP.npy', FP)
    # np.save('./save_weights/central.npy', central)
    np.save('./save_weights/full_power.npy', full_pwr)
    np.save('./save_weights/random_power.npy', random_pwr)
    np.save('./save_weights/centralized_DRL.npy', rewards)
    np.save('./save_weights/optimal_no_delay.npy', optimal_no_delay)
    np.save('./save_weights/optimal.npy', optimal)
    # np.save('./save_weights/multi_agent_DRL_test.npy', rewards)

    #plt.plot(dqn.loss)
    #plt.show()


class CustomQNetwork(q_network.QNetwork):
    def __init__(self,
                 input_tensor_spec,
                 action_spec,
                 fc_layer_params=(100, 50),
                 dropout_rate=0.5,
                 l2_reg=0.01,
                 activation_fn=tf.nn.relu,
                 **kwargs):  # Accept additional keyword arguments
        super(CustomQNetwork, self).__init__(
            input_tensor_spec,
            action_spec,
            fc_layer_params=fc_layer_params,
            activation_fn=activation_fn,
            **kwargs)  # Pass additional arguments to the superclass

        self._dropout_rate = dropout_rate
        self._l2_reg = l2_reg
        self._layers = []

        for num_units in fc_layer_params:
            self._layers.append(layers.Dense(
                num_units,
                activation=activation_fn,
                kernel_regularizer=regularizers.l2(l2_reg)))
            self._layers.append(layers.Dropout(dropout_rate))

        # Add the output layer
        num_actions = action_spec.maximum - action_spec.minimum + 1
        self._layers.append(layers.Dense(
            num_actions,
            activation=None))

    def call(self, inputs, step_type=None, network_state=(), training=False):
        del step_type  # unused
        x = tf.cast(inputs, tf.float32)
        for layer in self._layers:
            x = layer(x, training=training)
        return x, network_state

def main2():
    env_instance = DRLenv2()
    num_TTIs = env_instance.max_TTI
    num_simul_rounds = 10
    num_iterations = num_TTIs
    rewards = np.zeros((num_simul_rounds,num_iterations))

    optimal = np.zeros((num_simul_rounds, num_TTIs))
    optimal_no_delay = np.zeros((num_simul_rounds, num_TTIs))
    full_pwr = np.zeros((num_simul_rounds, num_TTIs))
    random_pwr = np.zeros((num_simul_rounds, num_TTIs))

    action_full_pwr = np.ones(env_instance.transmitters) * env_instance.pmax
    action_random = np.zeros(env_instance.transmitters)

    for episode in range(num_simul_rounds):

        train_env_raw = DRLenv2()
        eval_env_raw = DRLenv2()

        # Wrap these separate instances
        wrapped_train_env = suite_gym.wrap_env(train_env_raw)
        wrapped_eval_env = suite_gym.wrap_env(eval_env_raw)

        # Create the TensorFlow environments
        train_env = TFPyEnvironment(wrapped_train_env)
        eval_env = TFPyEnvironment(wrapped_eval_env)

        print("Action spec:", train_env.action_spec())
        print("Action spec shape:", train_env.action_spec().shape)



        training_losses = []
        validation_losses = []
        validation_interval = 10

        # 2. Agent Setup
        # Custom network architecture
        fc_layer_params = (1024, 512, 256)  # Adjusted layer sizes
        dropout_rate = 0  # Dropout rate (between 0 and 1)
        l2_reg = 0  # L2 regularization factor

        # Create the Q-Network
        q_net = CustomQNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=fc_layer_params,
            dropout_rate=dropout_rate,  # Pass the modified dropout rate
            l2_reg=l2_reg  # Pass the modified L2 regularization factor
        )


        # dropout_rate = dropout_rate,
        # l2_reg = l2_reg

        # Initial learning rate
        initial_learning_rate = 5e-3
        learning_rate_decay = 1e-4
        # global_step = tf.Variable(0, trainable=False)

        # Create a learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=5e-3,
            decay_steps=1,
            decay_rate=1 - 1e-4,
            staircase=False
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Train step counter
        train_step_counter = tf.Variable(0)

        discount_factor = 0.5

        total_iterations_between_updates = 100
        target_update_tau = 1

        # Create the DQN Agent
        agent = dqn_agent.DqnAgent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            q_network=q_net,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter,
            gamma=discount_factor,
            target_update_tau=target_update_tau,
            target_update_period=total_iterations_between_updates)

        agent.initialize()

        # 3. Replay Buffer
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=30000)

        # 4. Data Collection
        initial_epsilon = 0.2

        # Epsilon decay rate
        epsilon_decay = 1e-4

        # Minimum epsilon value
        min_epsilon = 0.01

        # Wrap the agent's policy with EpsilonGreedyPolicy
        epsilon_greedy_policy = eg_policy.EpsilonGreedyPolicy(
            agent.policy, epsilon=initial_epsilon)

        # Use this policy for data collection
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            train_env,
            epsilon_greedy_policy,
            observers=[replay_buffer.add_batch],
            num_steps=1)

        collect_steps_before_training = 50  # Collect this many steps before starting training
        for _ in range(collect_steps_before_training):
            collect_driver.run()

            # Early Stopping Parameters
        best_reward = -float('inf')
        patience = 50
        no_improvement_counter = 0
        early_stopped = False  # Flag to indicate if early stopping occurred

        # 5. Training Loop

        collect_steps_per_iteration = 1

        batch_size = 256
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)



        for iteration in range(num_iterations):
            # Collect a few steps and save to the replay buffer
            #print("iter =", iteration, train_env_raw.channel_gain)
            best, optimal_no_delay[episode, iteration] = find_optimal_actions(train_env_raw.channel_gain, env_instance.action_set,  env_instance.noise,  env_instance.transmitters)
            full_pwr[episode, iteration] = compute_sum_rate(train_env_raw.channel_gain, action_full_pwr,
                                                            env_instance.noise)

            # Generate random actions for the random power scheme
            for x in range(env_instance.transmitters):
                action_random[x] = env_instance.action_set[random.randint(0, env_instance.action_cand - 1)]

            random_pwr[episode, iteration] = compute_sum_rate(train_env_raw.channel_gain, action_random,
                                                              env_instance.noise)

            for _ in range(collect_steps_per_iteration):
                action_step = epsilon_greedy_policy.action(train_env.current_time_step())
                result = collect_driver.run()
                # print("TTI=", iteration, "Result:", result)  # Print the result to inspect its structure
                # print("TTI=", iteration, "Action:", action_step.action.numpy())

                # print("TTI=", iteration, "Reward:", result[0].reward.numpy())  # Print the reward
                #rewards(result[0].reward.numpy())  # Collect rewards
                rewards[episode, iteration] = result[0].reward.numpy().item()
                actions = env_instance.decode_action(action_step.action.numpy())


            # Sample a batch of data from the buffer and update the agent's network
            experience, unused_info = next(iterator)
            # print(experience)
            train_loss = agent.train(experience).loss
            # global_step.assign_add(1)

            # Update step counter, log loss, etc.
            step = agent.train_step_counter.numpy()
            #print("iter =", iteration, train_env_raw.channel_gain)



            training_losses.append(train_loss)

            # Update epsilon
            new_epsilon = max(min_epsilon, (1 - epsilon_decay) * epsilon_greedy_policy._epsilon)
            epsilon_greedy_policy._epsilon = new_epsilon

            # Optionally, log the current epsilon value
            if iteration % 100 == 0:
                print(f'Current epsilon: {epsilon_greedy_policy._epsilon}')


            if step % 100 == 0:
                print('step = {0}: state = {1}'.format(step, train_env_raw.state))
                print('step = {0}: channel gain = {1}'.format(step, train_env_raw.channel_gain))
                print('step = {0}: loss = {1}'.format(step, train_loss))
                print('step = {0}: DRL actions = {1}'.format(step, env_instance.decode_action(action_step.action.numpy())))
                #print('step = {0}: DRL actions = {1}'.format(step, env_instance.action_set[action_step.action.numpy()]))
                print('step = {0}: DRL reward = {1}'.format(step,rewards[episode, iteration]))
                print('step = {0}: Brute action = {1}'.format(step,best))
                print('step = {0}: Brute reward = {1}'.format(step, optimal_no_delay[episode, iteration]))
                print('step = {0}: Full reward = {1}'.format(step, full_pwr[episode, iteration]))
            '''
            if iteration % validation_interval == 0:
                episode_loss = 0.0
                num_time_steps = 0  # Initialize the number of time steps

                time_step = eval_env.reset()

                while not time_step.is_last():
                    action_step = agent.policy.action(time_step)
                    next_time_step = eval_env.step(action_step.action)

                    # Calculate the Q-values from the policy network
                    predicted_q_values, _ = agent._q_network(time_step.observation, time_step.step_type)
                    target_q_values, _ = agent._q_network(next_time_step.observation, next_time_step.step_type)

                    # Calculate the Q-value difference as the loss
                    immediate_loss = tf.reduce_mean(tf.square(predicted_q_values - target_q_values))

                    episode_loss += immediate_loss.numpy()
                    num_time_steps += 1  # Increment the number of time steps

                    time_step = next_time_step

                # Calculate the average validation loss over the episode and store it
                average_validation_loss = episode_loss / num_time_steps
                validation_losses.append(average_validation_loss)
                print("number of time step during ep = ", num_time_steps)
                print("validation_loss = ", average_validation_loss)
        
            '''
        train_env.reset()


    #np.save('./save_weights/centralized_DRL.npy', rewards)
    np.save('./save_weights/centralized_DRL_test2_large_node.npy', rewards)
    #np.save('./save_weights/full_power.npy', full_pwr)
    #np.save('./save_weights/random_power.npy', random_pwr)
    #np.save('./save_weights/optimal_no_delay.npy', optimal_no_delay)

    np.save('./save_weights/full_power_test2_large_node.npy', full_pwr)
    np.save('./save_weights/random_power_test2_large_node.npy', random_pwr)
    np.save('./save_weights/optimal_no_delay_test2_large_node.npy', optimal_no_delay)



    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Training')
    plt.legend()

    plt.show()


class CustomQNetwork(TFModelV2):
    """Custom model for Q-network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomQNetwork, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        # Define your layers here
        self.input_layer = tf.keras.layers.InputLayer(obs_space.shape)
        self.hidden_layers = [
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(40, activation="relu")
        ]
        self.output_layer = tf.keras.layers.Dense(num_outputs)

    def forward(self, input_dict, state, seq_lens):
        x = self.input_layer(input_dict["obs"])
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output, state


ModelCatalog.register_custom_model("custom_q_network", CustomQNetwork)


def exponential_lr_schedule(timestep):
    initial_learning_rate = 5e-3
    decay_rate = 1e-4
    return initial_learning_rate * (1 - decay_rate) ** (timestep / 3)

def custom_epsilon_decay(iteration, min_epsilon=0.01, initial_epsilon=0.2, decay_rate=1e-4):
    epsilon = initial_epsilon * (1 - decay_rate) ** iteration
    return max(epsilon, min_epsilon)


def main_multi2():
    register_env("DRLmultienv2", lambda _: DRLmultienv())
    env_instance = DRLmultienv()
    train_env_raw = DRLmultienv()
    eval_env_raw = DRLmultienv()

    ray.init()

    obs_space = train_env_raw.observation_space
    act_space = train_env_raw.action_space
    config = {
        "env": "DRLmultienv2",
        "multiagent": {
            "policies": {
                "shared_policy": (DQNTFPolicy, obs_space, act_space, {
                    "model": {
                        "custom_model": "custom_q_network",
                    },
                }),
            },
            "policy_mapping_fn": lambda agent_id: "shared_policy",
        },
        "lr_schedule": exponential_lr_schedule,
        "gamma": 0.5,
        "num_gpus": 0,
        "num_workers": env_instance.transmitters,
        "train_batch_size": 256,
        "replay_buffer_config": {
            "capacity": 3000,
            # Other replay buffer settings...
        },
        "rollout_fragment_length": 1,
        "train_batch_size": env_instance.transmitters,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 0.2,
            "final_epsilon": 0.01,
            "epsilon_timesteps": 20000,  # Timesteps over which to anneal epsilon
        },
        "target_network_update_freq": 100 * env_instance.transmitters
    }

    trainer = DQNTrainer(config=config)


    num_TTIs = env_instance.max_TTI
    num_simul_rounds = 1

    optimal = np.zeros((num_simul_rounds, num_TTIs))
    optimal_no_delay = np.zeros((num_simul_rounds, num_TTIs))
    full_pwr = np.zeros((num_simul_rounds, num_TTIs))
    random_pwr = np.zeros((num_simul_rounds, num_TTIs))

    # Wrap these separate instances


    action_full_pwr = np.ones(env_instance.transmitters) * env_instance.pmax
    action_random = np.zeros(env_instance.transmitters)

    training_losses = []
    validation_losses = []


    # 5. Training Loop
    num_iterations = num_TTIs
    rewards = np.zeros((num_simul_rounds, num_TTIs))

    for episode in range(num_simul_rounds):


        for iteration in range(num_iterations):
            # Collect a few steps and save to the replay buffer
            # print("iter =", iteration, train_env_raw.channel_gain)
            best, optimal_no_delay[episode, iteration] = find_optimal_actions(train_env_raw.channel_gain,
                                                                                   env_instance.action_set,
                                                                                   env_instance.noise,
                                                                                   env_instance.transmitters)
            full_pwr[episode, iteration] = compute_sum_rate(train_env_raw.channel_gain, action_full_pwr,
                                                            env_instance.noise)

            # Generate random actions for the random power scheme
            for x in range(env_instance.transmitters):
                action_random[x] = env_instance.action_set[random.randint(0, env_instance.action_cand - 1)]

            random_pwr[episode, iteration] = compute_sum_rate(train_env_raw.channel_gain, action_random,
                                                              env_instance.noise)
            actions = []

            result = trainer.train()

            loss = result['info']['learner']['default_policy']['learner_stats']['total_loss']
            training_losses.append(loss)

            rewards[episode, iteration] = train_env_raw.rates

            new_epsilon = custom_epsilon_decay(iteration)

            # Update the exploration epsilon of each policy
            for policy_id in trainer.workers.local_worker().policy_map:
                policy = trainer.get_policy(policy_id)
                policy.exploration.set_epsilon(new_epsilon)

            # Optionally, log the current epsilon value
            if iteration % 100 == 0:
                print(f'Current epsilon: {new_epsilon}')

            if iteration % 100 == 0:

                print('iteration = {0}: channel gain = {1}'.format(iteration, train_env_raw.channel_gain))
                print(f"Iteration: {iteration}, loss: {loss}")
                # print('iteration = {0}: DRL actions = {1}'.format(step, env_instance.decode_action(action_iteration.action.numpy())))
                #print('iteration = {0}: DRL actions = {1}'.format(iteration, env_instance.action_set[action_step.action.numpy()]))
                print('step = {0}: DRL reward = {1}'.format(iteration, rewards[episode, iteration]))
                print('step = {0}: Brute action = {1}'.format(iteration, best))
                print('step = {0}: Brute reward = {1}'.format(iteration, optimal_no_delay[episode, iteration]))
                print('step = {0}: Full reward = {1}'.format(iteration, full_pwr[episode, iteration]))

        train_env.reset()

    # rewards_array = np.array(rewards).reshape(-1, num_iterations)
    # np.save('./save_weights/centralized_DRL.npy', rewards)
    np.save('./save_weights/multi_agent_DRL.npy', rewards)
    np.save('./save_weights/full_power.npy', full_pwr)
    np.save('./save_weights/random_power.npy', random_pwr)
    np.save('./save_weights/optimal_no_delay.npy', optimal_no_delay)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()

    # Plot validation loss
    plt.subplot(1, 2, 2)
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Training')
    plt.legend()

    plt.show()







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


def find_optimal_actions(channel_gain, action_set, noise, num_agents):

    optimal_sum_rate = float('-inf')
    best_actions = [0] * num_agents

    # Iterate over all combinations of actions for each agent
    for actions in it.product(action_set, repeat=num_agents):
        current_sum_rate = compute_sum_rate(channel_gain, actions, noise)
        if current_sum_rate > optimal_sum_rate:
            optimal_sum_rate = current_sum_rate
            best_actions = actions

    return best_actions, optimal_sum_rate


def find_optimal_actions_same(channel_gain, action_set, noise, num_agents):
    optimal_sum_rate = float('-inf')
    best_action = 0

    # Iterate over each action in the action set
    for action in action_set:
        # Apply the same action to all agents
        actions = [action] * num_agents
        current_sum_rate = compute_sum_rate(channel_gain, actions, noise)
        if current_sum_rate > optimal_sum_rate:
            optimal_sum_rate = current_sum_rate
            best_action = action

    # The best actions are the same for all agents
    best_actions = [best_action] * num_agents

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
        rate = np.log(1 + SINR)
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
    centralized_DRL = np.load('./save_weights/centralized_DRL_test2_large_node.npy')
    multi_agent_DRL = np.load('./save_weights/multi_agent_DRL.npy')
    multi_agent_DRL_MIMO = np.load('./save_weights/multi_agent_DRL_MIMO.npy')
    FP = np.load('./save_weights/FP.npy')
    delayed_FP = np.load('./save_weights/central.npy')
    optimal = np.load('./save_weights/optimal.npy')
    optimal_no_delay = np.load('./save_weights/optimal_no_delay_test2_large_node.npy')
    full_pwr = np.load('./save_weights/full_power_test2_large_node.npy')
    random_pwr = np.load('./save_weights/random_power_test2_large_node.npy')
    rate_DRL = np.load('./save_weights/multi_agent_DRL_rate.npy')

    num_tx = 3
    num_simul_rounds = 10
    start = 20
    space = 250
    print(len(centralized_DRL))

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
    print(len(reward_avg))


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
        cumulative_rewards = moving_average(reward_avg, space)
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
        plt.plot(range(start, len(reward_avg)), cumulative_rewards, label='Centralized DRL')
        plt.plot(range(start, len(reward_avg_multi)), cumulative_rewards_multi, label='Multi-agent DRL')
        #plt.plot(range(start, len(reward_avg_multi)), cumulative_rate_multi, label='Multi-agent DRL')
        plt.plot(range(start, len(reward_avg_optimal_no_delay)), cumulative_rewards_optimal_no_delay,
                 label='Brute (no delay)')
        #plt.plot(range(start, len(reward_avg_optimal)), cumulative_rewards_optimal, label='Brute (delay)')
        plt.plot(range(start, len(reward_avg_full_pwr)), cumulative_rewards_full_pwr, label='Full power')
        plt.plot(range(start, len(reward_avg_random_pwr)), cumulative_rewards_random_pwr, label='Random power')
        #plt.plot(range(start, len(reward_avg_FP)), cumulative_rewards_FP, label='FP (no delay)')
        #plt.plot(range(start, len(reward_avg_delayed_FP)), cumulative_rewards_delayed_FP, label='Central (Delayed FP)')

        plt.legend()
        plt.xlabel('Time slot')
        plt.ylabel('Average of cumulative reward per link')

    if switch == 1:
        plt.plot(range(space - 1, space - 1 + len(cumulative_rewards)), cumulative_rewards,
                 label='Centralized DRL')
        plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_multi)), cumulative_rewards_multi, label='Multi-agent DRL')
        #plt.plot(range(space - 1, space - 1 + len(cumulative_rate_multi)), cumulative_rate_multi,
        #         label='Multi-agent DRL')
        #plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_optimal)), cumulative_rewards_optimal,
        #         label='Brute (delay)')
        plt.plot(range(space - 1, space - 1 + len(cumulative_rewards_optimal_no_delay)),
                 cumulative_rewards_optimal_no_delay, label='Brute (no delay)')
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
    #plt.yscale('log', base=10)
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
    f_d = 10
    T = 0.02
    rho = sp.jv(0, 2 * math.pi * f_d * T)
    print(rho)




if __name__ == "__main__":  ##인터프리터에서 실행할 때만 위 함수(main())을 실행해라. 즉, 다른데서 이 파일을 참조할 땐(import 시) def만 가져가고, 실행은 하지말라는 의미.
    # bitcheck()
    #main()
    #main2()
    #main_multi()
    main_multi2()
    # main_multi_MIMO()
    # opt()
    # fractional()
    #full_pwr()


    graph(0)
    #testing()
