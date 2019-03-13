import numpy as np


class TabularQLearningAlgorithm():

    def __init__(self, state_space_size, action_space_size, hashing_function, discount_factor, epsilon_greedy, learning_rate, kwargs):
        """
        TODO: Document
        """
        self.Q_table = np.zeros((state_space_size, action_space_size), dtype=np.float64)
        self.learning_rate = learning_rate
        self.hashing_function = hashing_function
        self.epsilon_greedy = epsilon_greedy
        self.discount_factor = discount_factor
        self.kwargs = kwargs
        assert learning_rate >= 0 and learning_rate <= 1, 'Learning ratev alue should be between [0,1]'
        assert epsilon_greedy >= 0 and epsilon_greedy <= 1, 'Epsilon greedy value should be between [0,1]'

    def update_q_table(self, s, a, r, succ_s):
        s, succ_s = self.hashing_function(s), self.hashing_function(succ_s)
        self.Q_table[s, a] += self.learning_rate * (r + self.discount_factor * max(self.Q_table[succ_s, :]) - self.Q_table[s, a])

    def find_moves(self, state, exploration):
        if exploration and np.random.uniform() <= self.epsilon_greedy:
            return np.random.choice(range(self.Q_table.shape[1]))

        state = self.hashing_function(state)
        optimal_moves = np.argwhere(self.Q_table[state, :] == np.amax(self.Q_table[state, :]))
        return np.random.choice(optimal_moves.flatten().tolist())
