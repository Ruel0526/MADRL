import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
#from typing import Dict
import numpy as np
import math
import random
import scipy.special as sp
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.envs.registration import register

from sklearn.preprocessing import MinMaxScaler

class DRLmultienv(MultiAgentEnv):
    """
    Custom Environment simulating wireless communication with transmitters and users.
    """

    def __init__(self):
        super(DRLmultienv, self).__init__()

        self.transmitters = 3  # Number of agents
        self._agent_ids = set([f'agent_{i}' for i in range(self.transmitters)])
        self.users = 3
        self.TTI = 0
        self.c = 2
        self.max_TTI = 5
        self.pmax = math.pow(10, 0.8)
        self.action_cand = 10
        self.action_set = np.linspace(0, self.pmax, self.action_cand)
        self.noise = math.pow(10, -14.4)

        self.state_size = 7+4*self.c+3*self.c

        self.f_d = 2
        self.T = 0.02
        self.rho = sp.jv(0, 2 * math.pi * self.f_d * self.T)

        # Define action and observation space for each agent
        self.action_space = Discrete(int(self.action_cand))
        '''
        self.observation_space = Dict({
            f'agent_{i}': Box(low=-np.inf, high=np.inf, shape=(self.state_size,),
                              dtype=np.float64)
            for i in range(self.transmitters)
        })
        '''
        self.observation_space = Box(low=np.float64(-np.inf), high=np.float64(np.inf), shape=(self.state_size,), dtype=np.float64)
        # Initialize state for each agent
        self.states = {f'agent_{i}': np.zeros(self.state_size) for i in range(self.transmitters)}
        self.old_states = {f'agent_{i}': np.zeros(self.state_size) for i in range(self.transmitters)}

        self.rates = 0
        self.SINR_cap = 10 ** (30 / 10)

        self.reset()

    def reset(self, *, seed=None, options=None):
        # Reset the state of the environment to an initial state
        self.TTI = 0
        self.rates = 0
        self.A = self.tx_positions_gen(self.transmitters, 100)
        self.B, self.inside_status = self.rx_positions_gen(self.A, 10, 100)
        self.H = np.ones((self.transmitters, self.transmitters)) * (
                random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        self.channel_gain = np.zeros((self.transmitters, self.users))
        for x in range(self.transmitters):
            for y in range(self.users):
                self.channel_gain[x, y] = self.channel_gain_function(self.A[x], self.B[y], self.H[x, y])

        # Reset state for each agent
        self.states = {f'agent_{i}': np.zeros(self.state_size) for i in range(self.transmitters)}

        # Return initial observation for each agent
        initial_observations = {
            f'agent_{i}': np.zeros(self.state_size, dtype=np.float64)
            for i in range(self.transmitters)
        }

        infos = {agent_id: {} for agent_id in initial_observations}  # Empty info dict for each agent
        return initial_observations, infos

    def action_space_sample(self, agent_ids=None):
        # If no specific agent_ids are provided, use all agents
        if agent_ids is None:
            agent_ids = self._agent_ids

        # Create a random action for each specified agent
        return {agent_id: self.action_space.sample() for agent_id in agent_ids}



    def tx_positions_gen(self, transmitter, R):
        tx_positions = []

        if transmitter == 19:
            tx_positions.append((-2 * R, 2 * R * math.sqrt(3)))
            tx_positions.append((0, 2 * R * math.sqrt(3)))
            tx_positions.append((2 * R, 2 * R * math.sqrt(3)))

            tx_positions.append((-3 * R, R * math.sqrt(3)))
            tx_positions.append((-R, R * math.sqrt(3)))
            tx_positions.append((R, R * math.sqrt(3)))
            tx_positions.append((3 * R, R * math.sqrt(3)))

            tx_positions.append((-4 * R, 0))
            tx_positions.append((-2 * R, 0))
            tx_positions.append((0, 0))
            tx_positions.append((2 * R, 0))
            tx_positions.append((4 * R, 0))

            tx_positions.append((-3 * R, -R * math.sqrt(3)))
            tx_positions.append((-R, -R * math.sqrt(3)))
            tx_positions.append((R, -R * math.sqrt(3)))
            tx_positions.append((3 * R, -R * math.sqrt(3)))

            tx_positions.append((-2 * R, -2 * R * math.sqrt(3)))
            tx_positions.append((0, -2 * R * math.sqrt(3)))
            tx_positions.append((2 * R, -2 * R * math.sqrt(3)))

        if transmitter == 3:
            tx_positions.append((-2 * R, 0))
            tx_positions.append((0, 0))
            tx_positions.append((-R, -R * math.sqrt(3)))

        return tx_positions

    def hexagon_vertices(self, center, edge_to_center_distance):
        """
        Calculate the vertices of a regular hexagon.
        Args:
        - center: Tuple (x, y) representing the center of the hexagon
        - edge_to_center_distance: Distance from the center to an edge of the hexagon
        Returns:
        - List of tuples representing the vertices of the hexagon
        """
        vertex_distance = edge_to_center_distance / math.cos(math.pi / 6)
        vertices = []
        for i in range(6):
            angle = (math.pi / 3) * i + (math.pi / 6)
            x = center[0] + vertex_distance * math.cos(angle)
            y = center[1] + vertex_distance * math.sin(angle)
            vertices.append((x, y))
        return vertices

    def is_inside_hexagon(self, x, y, vertices):
        """
        Check if a point is inside a regular hexagon defined by its vertices.
        Args:
        - x, y: Coordinates of the point
        - vertices: List of tuples representing the vertices of the hexagon
        Returns:
        - Boolean indicating whether the point is inside the hexagon
        """
        for i in range(6):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 6]
            edge_dir = (x2 - x1, y2 - y1)
            point_dir = (x - x1, y - y1)
            cross_product = edge_dir[0] * point_dir[1] - edge_dir[1] * point_dir[0]
            if cross_product <= 0:  # 외적이 0 이하인 경우 (왼쪽에 위치하지 않는 경우)
                return False
        return True

    def rx_positions_gen(self, tx_positions, r, R):
        rx_positions = []
        inside_status = []  # List to store if Rx is inside the hexagon
        for tx_pos in tx_positions:
            hex_vertices = self.hexagon_vertices(tx_pos, R)
            while True:
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(r, R / math.cos(math.pi / 6))
                x2 = tx_pos[0] + distance * math.cos(angle)
                y2 = tx_pos[1] + distance * math.sin(angle)
                if np.sqrt((x2 - tx_pos[0]) ** 2 + (y2 - tx_pos[1]) ** 2) >= r and self.is_inside_hexagon(x2, y2,
                                                                                                          hex_vertices):
                    rx_positions.append((x2, y2))
                    inside = self.is_inside_hexagon(x2, y2, hex_vertices)
                    inside_status.append(inside)
                    break
        return rx_positions, inside_status

    def channel_gain_function(self, tx_position, rx_position, small_scale):

        x1, y1 = tx_position
        x2, y2 = rx_position

        d_k = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        PL_0 = 120.9

        log_normal = 8
        pathloss = PL_0 + 37.6 * math.log10(d_k / 1000) + np.random.normal(0, log_normal)

        gain = small_scale.conjugate() * small_scale / (10 ** (pathloss / 10))

        return np.real(gain)

    def compute_sum_rate(self, channel_gain, actions, noise):
        sum_rate = 0
        sum_SINR = 0
        SINR_cap = 10 ** (30 / 10)
        for i in range(len(actions)):
            interferences = sum(channel_gain[j, i] * actions[j] for j in range(len(actions)) if j != i)
            SINR = channel_gain[i, i] * actions[i] / (interferences + noise)

            # Cap the SINR at 30 dB
            SINR = min(SINR, SINR_cap)
            # if SINR == SINR_cap:
            # print('It is over 30dB')

            sum_rate += math.log2(1 + SINR)
            sum_SINR += SINR
        return sum_rate, sum_SINR

    def compute_reward_for_agent(self, channel_gain, actions, noise, agent):

        SINR_cap = 10 ** (30 / 10)

        interferences = sum(channel_gain[j, agent] * actions[j] for j in range(len(actions)) if j != agent)
        SINR = channel_gain[agent, agent] * actions[agent] / (interferences + noise)

        # Cap the SINR at 30 dB
        SINR = min(SINR, SINR_cap)
        # if SINR == SINR_cap:
        # print('It is over 30dB')

        rate = math.log(1 + SINR)

        reward = rate

        for j in range(self.users):
            if j != agent:
                inter_of_interfered = np.sum(channel_gain[:, j] * actions) - channel_gain[j, j] * actions[j]
                SINR = channel_gain[j, j] * actions[j] / (inter_of_interfered + self.noise)
                SINR = min(SINR, self.SINR_cap)
                rate_with_agent = math.log(1 + SINR)
                inter_of_interfered_without_agent = inter_of_interfered - channel_gain[agent, j] * actions[agent]
                SINR = channel_gain[j, j] * actions[j] / (inter_of_interfered_without_agent + self.noise)
                SINR = min(SINR, self.SINR_cap)
                rate_without_agent = math.log(1 + SINR)
                reward -= (rate_without_agent - rate_with_agent)

        return rate, reward

    def decode_action(self, action):
        power_levels = []
        base = self.action_cand  # Assuming action_cand is the number of discrete levels per transmitter

        for _ in range(self.transmitters):
            level = action % base
            power = level * (self.pmax / (base - 1))
            power_levels.append(power)
            action //= base

        return power_levels

    def sort_and_select_top_c(self, agent, channel_gain, next_channel_gain, actions, c):
        interferer_gain = [(j, next_channel_gain[j, agent] * actions[agent]) for j in range(self.transmitters) if j != agent]
        interfered_gain = [(k, (channel_gain[agent, k] * actions[agent]) / (np.sum(channel_gain[:, k] * actions) - channel_gain[k, k] * actions[k] + self.noise)) for k in range(self.transmitters) if k != agent]

        # Sort based on gain and select top c
        top_c_interferers = sorted(interferer_gain, key=lambda x: x[1], reverse=True)[:c]
        top_c_interfered = sorted(interfered_gain, key=lambda x: x[1], reverse=True)[:c]

        # Sort the top c elements by j or k order
        top_c_interferers_sorted = sorted(top_c_interferers, key=lambda x: x[0])
        top_c_interfered_sorted = sorted(top_c_interfered, key=lambda x: x[0])

        return top_c_interferers_sorted, top_c_interfered_sorted

    def step(self, action_dict):
        done = self.TTI >= self.max_TTI
        rewards = {}
        infos = {}

        old_channel_gain = np.copy(self.channel_gain)

        print(action_dict)

        tx_powers = {agent_id: self.action_set[action] for agent_id, action in action_dict.items()}
        tx_power_list = [tx_powers[f'agent_{i}'] for i in range(self.transmitters)]
        tx_power_array = np.array(tx_power_list)

        self.rates = 0
        rate_array = np.zeros(self.transmitters)

        for agent_id, tx_power in tx_powers.items():
            agent = int(agent_id.split('_')[-1])

            rate, reward = self.compute_reward_for_agent(self.channel_gain, tx_power_array, self.noise, agent)

            self.rates += rate
            rate_array[agent] = rate

            rewards[agent_id] = reward

        for x in range(self.transmitters):
            for y in range(self.users):
                innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                htemp = self.rho * self.H[x, y] + (math.sqrt(1 - math.pow(self.rho, 2)) * innov)
                self.H[x, y] = htemp

        for x in range(self.transmitters):
            for y in range(self.users):
                self.channel_gain[x, y] = self.channel_gain_function(self.A[x], self.B[y], self.H[x, y])

        # Now compute the next state for each agent
        observations = {}
        terminateds = {}  # Change "dones" to "terminateds"
        truncateds = {}  # Add a "truncateds" dictionary

        for agent_id in action_dict:
            # Assuming agent_id is an integer like 0, 1, 2, etc.
            agent = int(agent_id.split('_')[-1])  # Extract agent number from agent_id
            old_state = self.old_states[agent_id]

            next_state = self.construct_observation_for_agent(agent, old_channel_gain, self.channel_gain, rate_array,
                                                              old_state, action_dict)

            #print("next_state", next_state)

            observations[agent_id] = next_state
            #print("observ", observations)

            terminateds[agent_id] = done  # Set to True if the episode is done
            truncateds[agent_id] = done  # Set to True if the episode is done

            self.old_states[agent_id] = next_state

        terminateds["__all__"] = done  # Set to True if the episode is done
        truncateds["__all__"] = done  # Set to True if the episode is done

        #print("Next obs:", observations)
        # Check if the observation is in the observation space
        '''
        for agent_id, obs in observations.items():
            if not self.observation_space[agent_id].contains(obs):
                print(f"Observation for {agent_id} is out of bounds:", obs)
        '''

        return observations, rewards, terminateds, truncateds, infos

    def construct_observation_for_agent(self, agent, channel_gain, next_channel_gain, rate_array, state, action_dict):


        # Initialize the next state array for this agent
        next_state = np.zeros(self.state_size)

        # Compute the next state for this agent
        actions = [self.action_set[action] for action in action_dict.values()]  # Convert actions to transmit power levels

        top_c_interferers, top_c_interfered = self.sort_and_select_top_c(agent, channel_gain, next_channel_gain,
                                                                         actions, self.c)
        #print(top_c_interferers)
        #print(top_c_interfered)

        next_state[0] = agent
        next_state[1] = actions[agent]
        next_state[2] = rate_array[agent]  # Assuming this is already computed
        next_state[3] = next_channel_gain[agent, agent]
        next_state[4] = channel_gain[agent, agent]
        next_state[5] = np.sum(next_channel_gain[:, agent] * actions) - next_channel_gain[agent, agent] * actions[agent] + self.noise
        next_state[6] = state[5]  # Assuming 'self.state' is the current state

        state_index = 7

        for j, _ in top_c_interferers:
            # Calculate interference from other agents
            # ...

            # Update the state array with interferer information
            next_state[state_index] = next_channel_gain[j, agent] * actions[
                j]  # self.normalize(next_channel_gain[j, agent] * actions[j], self.pmax)
            SINR_interferer = channel_gain[j, j] * actions[j] / (
                        np.sum(channel_gain[:, j] * actions) - channel_gain[j, j] * actions[j] + self.noise)
            SINR_interferer = min(SINR_interferer, self.SINR_cap)
            next_state[state_index + 1] = math.log(1 + SINR_interferer)

            next_state[state_index + 2] = state[state_index]

            next_state[state_index + 3] = state[state_index + 1]

            # Move to the next set of indices for the next interferer
            state_index += 4

        for k, _ in top_c_interfered:
            next_state[state_index] = channel_gain[k, k]

            SINR_interfered = channel_gain[k, k] * actions[k] / (
                        np.sum(channel_gain[:, k] * actions) - channel_gain[k, k] * actions[k] + self.noise)
            SINR_interfered = min(SINR_interfered, self.SINR_cap)

            next_state[state_index + 1] = math.log(1 + SINR_interfered)

            next_state[state_index + 2] = (channel_gain[agent, k] * actions[agent]) / (
                    np.sum(channel_gain[:, k] * actions) - channel_gain[k, k] * actions[k] + self.noise)

            state_index += 3

        return next_state
