import copy
from ..TQL import TabularQLearningAlgorithm
from ..TQL import RepeatedUpdateQLearningAlgorithm


class TabularQLearningAgent():
    def __init__(self, algorithm):
        self.name = 'TabularQLearning'
        self.training = True
        self.algorithm = algorithm

    def handle_experience(self, s, a, r, succ_s, done=False):
        if self.training:
            self.algorithm.update_q_table(s, a, r, succ_s)

    def take_action(self, state):
        return self.algorithm.find_moves(state, exploration=self.training)

    def clone(self, training=None):
        clone = copy.deepcopy(self)
        clone.training = training
        return clone


def build_TabularQ_Agent(task, config):
    state_space_size, action_space_size = task.state_space_size, task.action_dim
    hash_state = task.hash_function
    if config['use_repeated_update_q_learning']:
        algorithm = RepeatedUpdateQLearningAlgorithm(state_space_size, action_space_size, hash_state,
                                                     discount_factor=config['discount_factor'],
                                                     learning_rate=config['learning_rate'],
                                                     temperature=config['temperature'])
    else:
        algorithm = TabularQLearningAlgorithm(state_space_size, action_space_size, hash_state,
                                              discount_factor=config['discount_factor'],
                                              learning_rate=config['learning_rate'],
                                              epsilon_greedy=config['epsilon_greedy'])
    return TabularQLearningAgent(algorithm=algorithm)
