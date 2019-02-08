import numpy as np
import random


class TabularQLearningAlgorithm():

    def __init__(self, state_space_size, action_space_size, hashing_function, learning_rate=0.5, training=True):
        """
        TODO: Document
        """
        self.Q_table = np.zeros((state_space_size, action_space_size), dtype=np.float64)
        self.learning_rate = learning_rate
        self.hashing_function = hashing_function
        self.training = training
        self.name = 'TabularQLearning'
        pass

    def update_q_table(self, s, a, r, succ_s):
        self.Q_table[s, a] += self.learning_rate * (r + max(self.Q_table[succ_s, :]) - self.Q_table[s, a])

    def find_optimal_moves(self, Q_table, state):
        optimal_moves = np.argwhere(Q_table[state, :] == np.amax(Q_table[state, :]))
        return optimal_moves.flatten().tolist()
