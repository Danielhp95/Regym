import numpy as np


class RepeatedUpdateQLearningAlgorithm():
    '''
    Repeated Update Q Learning (RUQL) as introduced in:
    "Addressing the Policy Bias of Q-Learning by Repeating Updates" - Sherief Abdallah, Michael Kaisers
    '''
    def __init__(self, state_space_size, action_space_size, hashing_function, discount_factor, learning_rate, temperature, kwargs):
        self.Q_table = np.zeros((state_space_size, action_space_size), dtype=np.float64)
        self.learning_rate = learning_rate
        self.hashing_function = hashing_function
        self.temperature = temperature
        self.discount_factor = discount_factor
        self.kwargs = kwargs

    def update_q_table(self, s, a, r, succ_s):
        s, succ_s = self.hashing_function(s), self.hashing_function(succ_s)
        probability_taking_action_a = self.boltzman_exploratory_policy_from_state(s)[a]
        x = (1 - self.learning_rate)**(1 / probability_taking_action_a)
        self.Q_table[s, a] = x * self.Q_table[s, a] + (1 - x) * (r + self.discount_factor * max(self.Q_table[succ_s, :]))

    def boltzman_exploratory_policy_from_state(self, s):
        exp_q_values = np.exp([self.Q_table[s, i] / self.temperature for i in range(self.Q_table.shape[1])])
        normalizing_constant = sum(exp_q_values)
        return np.divide(exp_q_values, normalizing_constant)

    def find_moves(self, state, exploration):
        state = self.hashing_function(state)
        if exploration:
            p = self.boltzman_exploratory_policy_from_state(state)
            return np.random.choice(range(self.Q_table.shape[1]), p=p)
        else:
            optimal_moves = np.argwhere(self.Q_table[state, :] == np.amax(self.Q_table[state, :]))
            return np.random.choice(optimal_moves.flatten().tolist())
