import gym
import numpy as np
import math
import random
from gym import spaces
import scipy.special as sp

from sklearn.preprocessing import MinMaxScaler

import itertools as it

class DRLenv2(gym.Env):
    """
    Custom Environment simulating wireless communication with transmitters and users.
    """

    def __init__(self):
        super(DRLenv2, self).__init__()

        self.transmitters = 3
        self.users = 3
        self.TTI = 0
        self.max_TTI = 50000 # Define your max TTI
        #self.pmax = math.pow(10, 0.8)
        #self.pmax = 1
        self.action_cand = 10
        self.pmin = 1
        self.pmax = 1000
        #self.action_set = np.linspace(0, self.pmax, self.action_cand)

        min_power_dBm = 10 * np.log10(self.pmin)
        max_power_dBm = 10 * np.log10(self.pmax)
        step_size = (max_power_dBm - min_power_dBm) / 9
        discrete_power_dBm = np.arange(min_power_dBm, max_power_dBm + step_size, step_size)
        self.action_set = 10 ** (discrete_power_dBm / 10)


        self.noise = math.pow(10, -10.4) #mW
        #self.state_size = 2 * (1 + self.transmitters) + self.transmitters * self.users * 2
        #self.state_size = (2+ self.users*self.transmitters)*2
        #self.state_size = (5 + self.users * self.transmitters + self.users * self.transmitters) * 2
        self.state_size = (5 + self.users * self.transmitters)*2

        self.f_d = 10
        self.T = 0.02
        self.rho = sp.jv(0, 2 * math.pi * self.f_d * self.T)

        self.A = self.tx_positions_gen(self.transmitters, 500)
        self.B, self.inside_status = self.rx_positions_gen(self.A, 499, 500)

        # Define action and observation space
        self.action_space = spaces.Discrete(int(self.action_cand ** self.transmitters))
        #self.action_space = spaces.Discrete(int(self.action_cand ))
        self.observation_space = spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.state_size,), dtype=np.float32)

        # Initialize state
        self.state = None
        self.reset()

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.zeros(self.state_size)
        self.TTI = 0
        self.A = self.tx_positions_gen(self.transmitters, 100)
        #self.B, self.inside_status = self.rx_positions_gen(self.A, 99, 100)

        R = 100
        rx_positions = []

        '''
        rx_positions.append((-R + 3 * math.cos(5 * math.pi / 6), -R / math.sqrt(3) + 3 * math.sin(5 * math.pi / 6)))
        rx_positions.append((-R+3*math.cos(math.pi/6), -R / math.sqrt(3)+3*math.sin(math.pi/6)))
        rx_positions.append((-R+3*math.cos(3*math.pi/2), -R / math.sqrt(3)+3*math.sin(3*math.pi/2)))


        




        '''

        rx_positions.append((-2 * R+ (200/math.sqrt(3)-5) * math.cos(5 * math.pi / 6), 0+ (200/math.sqrt(3)-5) * math.sin(5 * math.pi / 6)))
        rx_positions.append((0+ (200/math.sqrt(3)-5) * math.cos(1 * math.pi / 6), 0+ (200/math.sqrt(3)-5) * math.sin(1 * math.pi / 6)))
        rx_positions.append((-R+ (200/math.sqrt(3)-5) * math.cos(3 * math.pi / 2), -R * math.sqrt(3) + (200/math.sqrt(3)-5) * math.sin(3 * math.pi / 2)))






        '''
        rx_positions.append((-2 * R + (90) * math.cos(0), 0 + (90) * math.sin(0)))
        rx_positions.append((0 + (90) * math.cos(math.pi), 0 + (90) * math.sin(math.pi)))
        rx_positions.append((-R+ (10) * math.cos(3 * math.pi / 2), -R * math.sqrt(3) + (10) * math.sin(3 * math.pi / 2)))


        '''
        
        '''

        rx_positions.append((-2 * R + (10) * math.cos(0), 0 + (10) * math.sin(0)))
        rx_positions.append((0 + (10) * math.cos(0), 0 + (10) * math.sin(0)))
        rx_positions.append(
            (-R + (90) * math.cos(3 * math.pi / 2), -R * math.sqrt(3) + (90) * math.sin(3 * math.pi / 2)))
        '''

        '''
        rx_positions.append((-2 * R + (30) * math.cos(0), 0 + (30) * math.sin(0)))
        rx_positions.append((0 + (50) * math.cos(4 * math.pi / 3), 0 + (50) * math.sin(4 * math.pi / 3)))
        rx_positions.append(
            (-R + (70) * math.cos(2 * math.pi / 3), -R * math.sqrt(3) + (70) * math.sin(2 * math.pi / 3)))
        '''




        self.B = rx_positions

        #self.H = np.ones((self.transmitters, self.transmitters)) * (
        #        random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        self.H = np.ones((self.transmitters, self.transmitters))
        self.channel_gain = np.zeros((self.transmitters, self.users))
        self.signal_powers = np.zeros((self.transmitters, self.users))
        for x in range(self.transmitters):
            for y in range(self.users):
                self.channel_gain[x, y] = self.channel_gain_function(self.A[x], self.B[y], self.H[x, y])
        # Initialize other necessary components
        # ...
        return self.state

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

        log_normal = 10
        #pathloss = PL_0 + 37.6 * math.log10(d_k / 1000) + np.random.normal(0, log_normal)
        pathloss = 140.7 + 36.7 * math.log10(d_k / 1000) #+ np.random.normal(0, log_normal)

        gain = small_scale.conjugate() * small_scale*0.1/ (10 ** (pathloss / 10))

        return np.real(gain)

    def compute_sum_rate(self, channel_gain, actions, noise):
        sum_rate = 0
        sum_SINR = 0
        SINR_cap = 10 ** (30 / 10)
        for i in range(len(actions)):
            interferences = sum(channel_gain[j, i] * actions[j] for j in range(len(actions)) if j != i)
            SINR = channel_gain[i, i] * actions[i] / (interferences + noise)

            # Cap the SINR at 30 dB
            #SINR = min(SINR, SINR_cap)
            # if SINR == SINR_cap:
            # print('It is over 30dB')

            sum_rate += math.log2(1 + SINR)
            sum_SINR += SINR
        return sum_rate

    def decode_action(self, action):
        #power_levels = []
        power_levels = np.zeros(self.transmitters)
        base = self.action_cand  # Assuming action_cand is the number of discrete levels per transmitter

        i=0
        for _ in range(self.transmitters):
            level = action % base
            #power = level * (self.pmax / (base - 1))
            power = self.action_set[int(level)]
            power_levels[i] = power
            i+=1
            action //= base

        return power_levels

    def step(self, action):

        #assert len(action) == self.transmitters, "Action vector must have the same length as the number of transmitters"
        # Implement the step function logic
        done = self.TTI >= self.max_TTI
        reward = 0
        info = {}
        #print("A =", self.A)
        #print("B =", self.B)

        old_state = np.copy(self.state)
        old_channel_gain = np.copy(self.channel_gain)
        old_signal_powers = np.copy(self.signal_powers)

        #actions = np.ones(self.transmitters) * self.action_set[action]
        actions = self.decode_action(action)

        reward = self.compute_sum_rate(self.channel_gain, actions, self.noise)
        best, guide_line = self.find_optimal_actions(self.channel_gain, self.action_set, self.noise, self.transmitters)


        for x in range(self.transmitters):
            for y in range(self.users):
                innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                htemp = self.rho * self.H[x, y] + (math.sqrt(1 - math.pow(self.rho, 2)) * innov)
                self.H[x, y] = htemp

        for x in range(self.transmitters):
            for y in range(self.users):
                self.channel_gain[x, y] = self.channel_gain_function(self.A[x], self.B[y], self.H[x, y])
                self.signal_powers[x, y] = self.channel_gain[x, y] * actions[x]



        #print("channel gain =" ,self.channel_gain)

        scaler = MinMaxScaler()
        gain_to_state = np.copy(self.channel_gain)
        channel_gain_reshaped = gain_to_state.reshape(-1, 1)
        scaled_channel_gain = scaler.fit_transform(channel_gain_reshaped)
        scaled_channel_gain = scaled_channel_gain.reshape(self.channel_gain.shape).flatten()
        signals_to_state = np.copy(self.signal_powers)
        signals_reshaped = signals_to_state.reshape(-1, 1)
        scaled_signals = scaler.fit_transform(signals_reshaped)
        scaled_signals = scaled_signals.reshape(self.signal_powers.shape).flatten()

        old_channel_gain_reshaped = old_channel_gain.reshape(-1, 1)
        old_scaled_channel_gain = scaler.fit_transform(old_channel_gain_reshaped)
        old_scaled_channel_gain = old_scaled_channel_gain.reshape(self.channel_gain.shape).flatten()

        old_signal_powers_reshaped = old_signal_powers.reshape(-1, 1)
        old_scaled_signal_powers = scaler.fit_transform(old_signal_powers_reshaped)
        old_scaled_signal_powers = old_scaled_signal_powers.reshape(self.signal_powers.shape).flatten()

        '''
        h_mean = np.mean(self.channel_gain)
        h_std = np.std(self.channel_gain)
        scaled_channel_gain = (self.channel_gain-h_mean)/h_std
        old_h_mean = np.mean(old_channel_gain)
        old_h_std = np.std(old_channel_gain)
        old_scaled_channel_gain = (old_channel_gain - old_h_mean) / old_h_std
        '''
        '''
        m, n = self.channel_gain.shape
        self.state[0] = action
        self.state[1] = reward
        state_index = 2
        self.state[state_index:state_index + m * n] = scaled_channel_gain
        self.state[state_index + m * n] = old_state[0]
        self.state[state_index + m * n + 1] = old_state[1]
        self.state[state_index + m * n + 2:] = old_scaled_channel_gain
        '''
        '''
        m, n = self.channel_gain.shape
        self.state[0] = action
        self.state[1] = actions[0]
        self.state[2] = actions[1]
        self.state[3] = actions[2]
        self.state[4] = reward
        state_index = 5
        self.state[state_index:state_index + m * n] = scaled_channel_gain
        self.state[state_index + m * n:state_index + m * n + m * n] = scaled_signals
        self.state[state_index + m * n + m * n] = old_state[0]
        self.state[state_index + m * n + m * n + 1] = old_state[1]
        self.state[state_index + m * n + m * n + 2] = old_state[2]
        self.state[state_index + m * n + m * n + 3] = old_state[3]
        self.state[state_index + m * n + m * n + 4] = old_state[4]
        self.state[state_index + m * n + m * n + 5: state_index + m * n + m * n + 5 + m * n] = old_scaled_channel_gain
        self.state[state_index + m * n + m * n + 5 + m * n: state_index + m * n + m * n + 5 + m * n + m * n] = old_scaled_signal_powers
        '''

        '''
        m, n = self.channel_gain.shape
        self.state[0] = action
        self.state[1] = actions[0]
        self.state[2] = actions[1]
        self.state[3] = actions[2]
        self.state[4] = reward
        state_index = 5
        self.state[state_index:state_index + m * n] = self.channel_gain.flatten()

        self.state[state_index + m * n] = old_state[0]
        self.state[state_index + m * n+1] = old_state[1]
        self.state[state_index + m * n+2] = old_state[2]
        self.state[state_index + m * n+3] = old_state[3]
        self.state[state_index + m * n+4] = old_state[4]

        self.state[state_index + m * n+5:] = old_channel_gain.flatten()
        '''

        scaled_power = []
        for x in range(self.transmitters):
            scaled_power.append(actions[x]/self.pmax)

        m, n = self.channel_gain.shape
        self.state[0] = action/(self.action_cand ** self.transmitters)
        self.state[1] = scaled_power[0]
        self.state[2] = scaled_power[1]
        self.state[3] = scaled_power[2]

        self.state[4] = reward#/guide_line
        state_index = 5
        self.state[state_index:state_index + m * n] = scaled_channel_gain

        self.state[state_index + m * n] = old_state[0]
        self.state[state_index + m * n + 1] = old_state[1]
        self.state[state_index + m * n + 2] = old_state[2]
        self.state[state_index + m * n + 3] = old_state[3]
        self.state[state_index + m * n + 4] = old_state[4]

        self.state[state_index + m * n + 5:] = old_scaled_channel_gain



        return self.state, reward, done, info

    # Define other necessary methods and functions
    # ...

    def find_optimal_actions(self, channel_gain, action_set, noise, num_agents):

        optimal_sum_rate = float('-inf')
        best_actions = [0] * num_agents

        # Iterate over all combinations of actions for each agent
        for actions in it.product(action_set, repeat=num_agents):
            current_sum_rate = self.compute_sum_rate(channel_gain, actions, noise)
            if current_sum_rate > optimal_sum_rate:
                optimal_sum_rate = current_sum_rate
                best_actions = actions

        return best_actions, optimal_sum_rate
