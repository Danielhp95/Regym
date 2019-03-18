import numpy as np

class MixedStrategyAgent():
    '''
    Representes a fixed agent which uses a mixed strategy.
    A mixed strategy is represented as a support vector (probability) distribution over
    the set of all possible actions.
    '''

    def __init__(self, support_vector, name):
        '''
        Checks that the support vector is a valid probability distribution
        :param support_vector: support vector for all three possible pure strategies [ROCK, PAPER, SCISSORS]
        :throws ValueError: If support vector is not a valid probability distribution
        '''
        if any(map(lambda support: support < 0, support_vector)):
            raise ValueError('Every support in the support vector should be a positive number. Given supports: {}'.format(support_vector))
        if sum(support_vector) != 1.0:
            raise ValueError('The sum of all supports in the support_vector should sum up to 1. Given supports: {}'.format(support_vector))
        self.support_vector = support_vector
        self.name = name
        self.nbr_actor = 1

    def set_nbr_actor(self, nbr_actor):
        self.nbr_actor = nbr_actor

    def take_action(self, state):
        '''
        Samples an action based on the probabilities presented by the agent's support vector
        :param state: Ignored for fixed agents
        '''
        return np.asarray([np.random.choice([0, 1, 2], p=self.support_vector) for _ in range(self.nbr_actor)]).reshape( (self.nbr_actor,-1))
        
    def handle_experience(self, *args):
        pass

    def clone(self, training=None, path=None):
        return MixedStrategyAgent(support_vector=self.support_vector, name=self.name)
