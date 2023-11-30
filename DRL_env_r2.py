import gym
import numpy as np
import math
import random
from gym import spaces
import scipy.special as sp

class DRLenv2(gym.Env):
    """
    Custom Environment simulating wireless communication with transmitters and users.
    """

    def __init__(self):
        super(DRLenv2, self).__init__()

        self.transmitters = 3
        self.users = 3
        self.TTI = 0
        self.max_TTI = 40000 # Define your max TTI
        self.pmax = math.pow(10, 0.8)
        self.action_cand = 2
        self.action_set = np.linspace(0, self.pmax, self.action_cand)
        self.noise = math.pow(10, -14.4)
        self.state_size = 2 * (1 + self.transmitters) + self.transmitters * self.users * 2

        self.f_d = 10
        self.T = 0.02
        self.rho = sp.jv(0, 2 * math.pi * self.f_d * self.T)

        self.A = self.tx_positions_gen(self.transmitters, 100)
        self.B, self.inside_status = self.rx_positions_gen(self.A, 10, 100)

        # Define action and observation space
        self.action_space = spaces.Discrete(1000)
        self.observation_space = spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(self.state_size,), dtype=np.float32)

        # Initialize state
        self.state = None
        self.reset()

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.zeros(self.state_size)
        self.TTI = 0
        self.A = self.tx_positions_gen(self.transmitters, 100)
        self.B, self.inside_status = self.rx_positions_gen(self.A, 10, 100)
        self.H = np.ones((self.transmitters, self.transmitters)) * (
                random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j)
        self.channel_gain = np.zeros((self.transmitters, self.users))
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

        log_normal = 8
        pathloss = PL_0 + 37.6 * math.log10(d_k / 1000) + np.random.normal(0, log_normal)

        gain = small_scale.conjugate() * small_scale / (10 ** (pathloss / 10))

        return np.real(gain)

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

    def decode_action(self, action):
        power_levels = []
        base = self.action_cand  # Assuming action_cand is the number of discrete levels per transmitter

        for _ in range(self.transmitters):
            level = action % base
            power = level * (self.pmax / (base - 1))
            power_levels.append(power)
            action //= base

        return power_levels

    def step(self, action):

        #assert len(action) == self.transmitters, "Action vector must have the same length as the number of transmitters"
        # Implement the step function logic
        done = self.TTI >= self.max_TTI
        reward = 0
        info = {}

        old_state = np.copy(self.state)
        old_channel_gain = np.copy(self.channel_gain)

        actions = self.decode_action(action)

        reward = self.compute_sum_rate(self.channel_gain, actions, self.noise)

        for x in range(self.transmitters):
            for y in range(self.users):
                innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j
                htemp = self.rho * self.H[x, y] + (math.sqrt(1 - math.pow(self.rho, 2)) * innov)
                self.H[x, y] = htemp

        for x in range(self.transmitters):
            for y in range(self.users):
                self.channel_gain[x, y] = self.channel_gain_function(self.A[x], self.B[y], self.H[x, y])

        m, n = self.channel_gain.shape
        required_size = 4 + (m * n) * 2
        if self.state_size < required_size:
            raise ValueError(f"self.state size ({self.state_size}) is smaller than required ({required_size})")

        self.state[0] = reward
        state_index = 1
        for i in range(self.transmitters):
            self.state[state_index] = actions[i]
            state_index += 1
        for i in range(self.transmitters + 1):
            self.state[state_index] = old_state[state_index - self.transmitters - 1]
            state_index += 1

        self.state[state_index:state_index + m * n] = self.channel_gain.flatten()
        self.state[state_index + m * n:] = old_channel_gain.flatten()

        self.TTI += 1

        del old_channel_gain
        del old_state

        return self.state, reward, done, info

    # Define other necessary methods and functions
    # ...
