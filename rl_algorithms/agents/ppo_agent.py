#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..PPO import PPOAlgorithm
from ..component import *


class PPOAgent():

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def handle_experience(self, s, a, r, succ_s, done=False):
        raise NotImplementedError('Not yet implemented')

    def take_action(self, state):
        raise NotImplementedError('Not yet implemented')

    def clone(self, training=None, path=None):
        from ..agent_hook import AgentHook
        cloned = AgentHook(self, training=training, path=path)
        return cloned


def build_PPO_Agent(task, config):
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for ppo agent
    :returns: PPOAgent adapted to be trained on :param: task under :param: config
    '''
    kwargs = dict()
    kwargs['model'] = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                                NatureConvBody())
    ppo_algorithm = PPOAlgorithm(kwargs)
    return PPOAgent(algorithm=ppo_algorithm)
