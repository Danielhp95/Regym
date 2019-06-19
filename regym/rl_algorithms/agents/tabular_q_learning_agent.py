import copy
import numpy as np 
from ..TQL import TabularQLearningAlgorithm
from ..TQL import RepeatedUpdateQLearningAlgorithm


class TabularQLearningAgent():
    def __init__(self, name, algorithm):
        self.name = name
        self.training = True
        self.algorithm = algorithm
        self.nbr_actor = self.algorithm.config['nbr_actor']

    def set_nbr_actor(self, nbr_actor):
        self.nbr_actor = nbr_actor
        self.algorithm.kwargs['nbr_actor'] = nbr_actor

    def handle_experience(self, s, a, r, succ_s, done=False):
        if self.training:
            self.algorithm.update_q_table(s, a, r, succ_s)

    def take_action(self, state):
        assert( len(state) == self.nbr_actor)
        return [ self.algorithm.find_moves(s, exploration=self.training) for s in state]
        
    def clone(self, training=None):
        clone = copy.deepcopy(self)
        clone.training = training
        return clone


def build_TabularQ_Agent(task, config, agent_name):
    state_space_size, action_space_size = task.state_space_size, task.action_dim
    hash_state = task.hash_function
    if config['use_repeated_update_q_learning']:
        algorithm = RepeatedUpdateQLearningAlgorithm(state_space_size, action_space_size, hash_state,
                                                     discount_factor=config['discount_factor'],
                                                     learning_rate=config['learning_rate'],
                                                     temperature=config['temperature'],
                                                     kwargs=config)
    else:
        algorithm = TabularQLearningAlgorithm(state_space_size, action_space_size, hash_state,
                                              discount_factor=config['discount_factor'],
                                              learning_rate=config['learning_rate'],
                                              epsilon_greedy=config['epsilon_greedy'],
                                              kwargs=config)
    return TabularQLearningAgent(name=agent_name, algorithm=algorithm)
