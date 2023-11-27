import numpy as np
import random
import math
import scipy.special as sp


class DRLenv(object):
    def __init__(self):


        self.transmitters = 19
        #self.tx_position = self.tx_positions_gen()


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
    '''
    def rx_positions_gen(self, tx_position):
        rx_positions=[]
        r = 10
        R = 500
        for tx_pos in tx_position:
            x1, y1 = tx_pos
            while True:
                x2 = random.uniform(x1 - R, x1 + R)
                y2 = random.uniform(y1 - R, y1 + R)
                if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) >= r:
                    rx_positions.append((x2, y2))
                    break
        return rx_positions
        
    '''


    '''
    def rx_positions_gen(self, tx_positions, r, R):
        """
        Generate receiver positions randomly within each hexagonal cell but outside the inner region.
        Args:
        - tx_positions: List of transmitter positions
        - r: Radius of the inner region where no receiver is placed
        - R: Cell radius (distance to hexagon vertex)
        Returns:
        - List of tuples representing receiver positions
        """
        rx_positions = []
        effective_radius = R   # Reduce the radius to decrease the chance of being outside the hexagon
        for tx_pos in tx_positions:
            x1, y1 = tx_pos
            while True:
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(r, effective_radius)
                x2 = x1 + distance * math.cos(angle)
                y2 = y1 + distance * math.sin(angle)
                if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) >= r:
                    rx_positions.append((x2, y2))
                    break
        return rx_positions
    '''

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
            if cross_product <= 0:  #외적이 0 이하인 경우 (왼쪽에 위치하지 않는 경우)
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
                if np.sqrt((x2 - tx_pos[0]) ** 2 + (y2 - tx_pos[1]) ** 2) >= r and self.is_inside_hexagon(x2, y2, hex_vertices):
                    rx_positions.append((x2, y2))
                    inside = self.is_inside_hexagon(x2, y2, hex_vertices)
                    inside_status.append(inside)
                    break
        return rx_positions, inside_status



    def Jakes_channel(self, previous_channel): # 유저 한명의 채널

        f_d = 10
        T = 0.02
        rho = sp.jv(0, 2*math.pi*f_d*T)


        initial = np.zeros(10)



        innov = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j

            #if previous_channel == initial:
            #    h = random.gauss(0, np.sqrt(1 / 2)) + random.gauss(0, np.sqrt(1 / 2)) * 1j



        h = rho * previous_channel + (math.sqrt(1-math.pow(rho, 2)) * innov)

            #channel = math.pow(np.absolute(h), 2) * pathloss

        channel_vector = h

        return channel_vector


    def channel_gain(self,  tx_position, rx_position, small_scale):

        x1, y1 = tx_position
        x2, y2 = rx_position

        d_k = np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
        PL_0 = 120.9

        log_normal = 8
        pathloss = PL_0 + 37.6 * math.log10(d_k/1000) + np.random.normal(0, log_normal)

        gain = small_scale.conjugate() * small_scale / (10 ** (pathloss / 10))

        return np.real(gain)

