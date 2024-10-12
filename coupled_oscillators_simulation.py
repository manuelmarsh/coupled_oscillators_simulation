# -*- coding: utf-8 -*-
"""
@author: Manuel Martini
"""

"""
This script simulates a system of oscillators using different representations
(adjacency matrix, adjacency list, incidence list) and solves the system using
various numerical methods (Euler Forward, Euler Backward, and Crank-Nicholson).
It generates an animation to visualize the simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Base class for the oscillators system
class OscillatorSystem:
    def __init__(self, fixed_indices, masses, positions):
        """
        fixed_indices: array with indices of fixed vertices
        masses: array of length n with the masses of the vertices
        positions: nx2 matrix with the initial positions of the vertices
        """
        self.fixed_indices = fixed_indices
        self.masses = masses 
        self.positions = positions
        self.n_vertices = len(positions[:, 0])  # Number of vertices

    def get_matrix(self):
        raise NotImplementedError  # To be implemented in subclasses

# Subclass using adjacency matrix representation
class AdjacencyMatrixSystem(OscillatorSystem):
    def __init__(self, K, fixed_indices, masses, positions):
        """
        K: nxn matrix where aij=kij, the elastic constant connecting two vertices.
        """
        OscillatorSystem.__init__(self, fixed_indices, masses, positions)
        self.constants = K

    def get_matrix(self):
        return self.constants / self.masses[:, None]  # Return matrix with aij = kij/mi

# Subclass using adjacency list representation
class AdjacencyListSystem(OscillatorSystem):
    def __init__(self, adjacency_list, fixed_indices, masses, positions):
        """
        adjacency_list: list of neighbors and connection constants
        Format: [[(v00,k00), (v02, k02), ..], [(v1,k1), ...]]
        """
        OscillatorSystem.__init__(self, fixed_indices, masses, positions)
        self.adjacency_list = adjacency_list

    def get_matrix(self):
        mat = np.zeros(shape=(self.n_vertices, self.n_vertices))
        for i in range(self.n_vertices):
            neighbors = self.adjacency_list[i]
            for neighbor in neighbors:
                j = neighbor[0]  # Index of the connected vertex
                k = neighbor[1]  # Elastic constant
                mat[i, j] = k
        return mat / self.masses[:, None]

# Subclass using incidence list representation
class IncidenceListSystem(OscillatorSystem):
    def __init__(self, incidence_list, fixed_indices, masses, positions):
        """
        incidence_list: list of edges represented as (i, j, k) tuples, 
        where i and j are the vertices and k is the elastic constant.
        """
        OscillatorSystem.__init__(self, fixed_indices, masses, positions)
        self.incidence_list = incidence_list

    def get_matrix(self):
        mat = np.zeros(shape=(self.n_vertices, self.n_vertices))
        for edge in self.incidence_list:
            i, j, k = edge
            mat[i, j] = k
            mat[j, i] = k
        return mat / self.masses[:, None]

def setup_approximation_method(system, t, initial_velocities):
    """
    system: instance of OscillatorSystem
    t: time array
    initial_velocities: nx2 matrix of initial velocities
    """
    K_M = system.get_matrix()  # Matrix where aij = kij/mi
    h = t[1] - t[0]  # Time step
    n = system.n_vertices
    m = t.size  # Number of time steps
    sequence = np.zeros(shape=(2*n, 2, m))  # Store positions and velocities
    sequence[0:n, :, 0] = system.positions
    sequence[n:2*n, :, 0] = initial_velocities
    # Define blocks of the A matrix for time evolution
    quad_1 = np.identity(n)
    quad_2 = np.zeros(shape=(n, n))
    quad_3 = K_M - np.diag(np.sum(K_M, axis=1))  # Matrix for the system dynamics
    quad_4 = np.zeros(shape=(n, n))
    A = np.vstack((np.hstack((quad_2, quad_1)), np.hstack((quad_3, quad_4))))  # System matrix
    return n, m, h, A, sequence

def forward_euler(system, t, initial_velocities):
    """ Forward Euler method for the simulation. Returns an array (n, 2, m). """
    n, m, h, A, sequence = setup_approximation_method(system, t, initial_velocities)
    for z in range(1, m):
        sequence[:, :, z] = h * A @ sequence[:, :, z-1] + sequence[:, :, z-1]
        for i in system.fixed_indices:  # Ensure fixed vertices remain at rest
            sequence[n+i, :, z] = np.array((0, 0))
    return sequence[0:n]

def backward_euler(system, t, initial_velocities, error=1e-7):
    """ Backward Euler method with an iterative approach. """
    n, m, h, A, sequence = setup_approximation_method(system, t, initial_velocities)
    r_before = sequence[:, :, 0]
    for z in range(1, m):
        f = lambda r: h * A @ r + sequence[:, :, z-1]
        r_next = f(r_before)
        for i in system.fixed_indices:  # Ensure fixed vertices remain at rest
            r_next[n+i, :] = np.array((0, 0))
        while np.linalg.norm(r_next - r_before) > error:
            r_before = r_next.copy()
            r_next = f(r_next)
            for i in system.fixed_indices:
                r_next[n+i, :] = np.array((0, 0))
        sequence[:, :, z] = r_next
    return sequence[0:n]

def crank_nicholson(system, t, initial_velocities, error=1e-7):
    """ Crank-Nicholson method for more accurate simulation. """
    n, m, h, A, sequence = setup_approximation_method(system, t, initial_velocities)
    r_before = sequence[:, :, 0]
    for z in range(1, m):
        f = lambda r: h * (A @ r + A @ sequence[:, :, z-1]) / 2 + sequence[:, :, z-1]
        r_next = f(r_before)
        for i in system.fixed_indices:  # Ensure fixed vertices remain at rest
            r_next[n+i, :] = np.array((0, 0))
        while np.linalg.norm(r_next - r_before) > error:
            r_before = r_next.copy()
            r_next = f(r_next)
            for i in system.fixed_indices:
                r_next[n+i, :] = np.array((0, 0))
        sequence[:, :, z] = r_next
    return sequence[0:n]

if __name__ == '__main__':
    # Common data
    fixed_indices = np.array([0])
    masses = np.array([1, 4, 1])
    initial_positions = np.array([(1, 3), (5, 4), (4, 3)])
    initial_velocities = np.array([(0, 0), (0, 0), (0, 0)])

    # Different representations:
    adjacency_matrix = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]])
    adjacency_list = [[(1, 3), (2, 1)], [(0, 3), (2, 2)], [(0, 1), (1, 2)]]
    incidence_list = [(0, 1, 3), (1, 2, 2), (2, 0, 1)]

    # Create system objects
    as_matrix = AdjacencyMatrixSystem(adjacency_matrix, fixed_indices, masses, initial_positions)
    as_adj_list = AdjacencyListSystem(adjacency_list, fixed_indices, masses, initial_positions)
    as_inc_list = IncidenceListSystem(incidence_list, fixed_indices, masses, initial_positions)

    t = np.linspace(0, 10, num=200)
    result = backward_euler(as_matrix, t, initial_velocities)  # Use backward Euler for simulation
    x = np.squeeze(result[:, 0, :])  # Reduce to matrix of shape (n_vertices, n_time_intervals)
    y = np.squeeze(result[:, 1, :])

    # Initialize figure and axis for the animation:
    fig, ax = plt.subplots()
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    
    # Initialize the scatter object with initial positions of the oscillators
    scat = ax.scatter(x[:, 0], y[:, 0], s=50, c='blue', marker='o')  # Start with the first positions

    # Update function for the animation at each time step
    def update(i):
        # Update scatter offsets to current position at time step i
        scat.set_offsets(np.c_[x[:, i], y[:, i]])  # Show all vertices at each time step
        return scat,

    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

    # Save the animation as a GIF
    plt.show()
