import gym
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


def obtain_observation_dimensions(env):
    '''
    Finds the dimensions of the action space for the agent on a symmetric multiagent environment.
    :param env: multiagent environment. Observation space symmetry is assumed. All agents share the same observation space
    :return: observation dimensions
    '''
    if not hasattr(env.observation_space, 'spaces') or len(env.observation_space.spaces) <= 1:
        raise ValueError("Environment is not multiagent")
    return int(np.prod(env.observation_space.spaces[0].shape))


def obtain_action_dimensions(env):
    '''
    Finds the dimensions of the action space for the agent on a symmetric multiagent environment.
    If the action space is discrete, return action count
    If the action space is multidimensional, return shape of action space
    :param env: multiagent environment. Action space symmetry is assumed. All agents share the same observation space
    :return: action dimensions
    '''
    if isinstance(env.action_space.spaces[0], Discrete): return env.action_space.spaces[0].n
    elif isinstance(env.action_space.spaces[0], Box): return env.action_space.spaces[0].shape[0]
    else: assert 'unknown action space'
