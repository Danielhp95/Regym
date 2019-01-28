import numpy as np
import random
import copy

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

    def clone(self, training=False):
        cloned = TabularQLearning(self.Q_table.shape[0], self.Q_table.shape[1], self.hashing_function,
                                  learning_rate=self.learning_rate, training=training)
        cloned.Q_table = copy.deepcopy(self.Q_table)
        return cloned

    
class TabularQLearningAgent(TabularQLearningAlgorithm):
    def __init__(self, state_space_size, action_space_size, hashing_function, learning_rate=0.5, training=True):
        """
        TODO: 
        Document
        Remove clone4 + queue2
        """
        self.Q_table = np.zeros((state_space_size, action_space_size), dtype=np.float64)
        self.learning_rate = learning_rate
        self.hashing_function = hashing_function
        self.training = training
        self.name = 'TabularQLearning'
        pass

    def handle_experience(self, s, a, r, succ_s, done=False):
        if self.training:
            self.update_q_table(self.hashing_function(s), a, r, self.hashing_function(succ_s))
            self.anneal_learning_rate()

    def take_action(self, state):
        optimal_moves = self.find_optimal_moves(self.Q_table, self.hashing_function(state))
        return random.choice(optimal_moves)

    def anneal_learning_rate(self):
        pass

    def clone(self, training=None, path=None):
        from .interface import AgentHook
        cloned = AgentHook(self,training=training, path=path)
        return cloned

def build_TabularQ_Agent(state_space_size,
                            action_space_size,
                            hash_state
                            ) :
    return TabularQLearningAgent(state_space_size, action_space_size, hash_state)

