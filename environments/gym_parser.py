import gym
import gym_rock_paper_scissors
from gym_rock_paper_scissors.envs.one_hot_space import OneHotEncoding
import numpy as np
from gym.spaces import Box, Discrete, Tuple

from collections import namedtuple

Task = namedtuple('Task', 'name observation_dim observation_type action_dim action_type hash_function')


def get_observation_dimensions_and_type(env):
    def parse_dimension_space(space):
        if isinstance(space, OneHotEncoding): return space.size, 'Discrete'
        if isinstance(space, Discrete): return space.n, 'Discrete'
        if isinstance(space, Box): return int(np.prod(space.shape)), 'Continuous'
        if isinstance(space, Tuple): return sum([parse_dimension_space(s)[0] for s in space.spaces]), parse_dimension_space(space.spaces[0])[1]
        raise ValueError('Unknown observation space: {}'.format(space))

    if not hasattr(env.observation_space, 'spaces') or len(env.observation_space.spaces) <= 1:
        raise ValueError("Environment is not multiagent")
    return parse_dimension_space(env.observation_space.spaces[0])


def get_action_dimensions_and_type(env):
    space = env.action_space.spaces[0]
    if isinstance(space, Discrete): return space.n, 'Discrete'
    elif isinstance(space, Box): return space.shape[0], 'Continuous'
    else: raise ValueError('Unknown action space: {}'.format(space))


def parse_gym_environment(env):
    '''
    Assumptions:
        - Observation / Action space (it's geometry, dimensionality) is identical for all agents
    :param env: multiagent environment.
    '''
    name = env.spec.id
    action_dims, action_type = get_action_dimensions_and_type(env)
    observation_dims, observation_type = get_observation_dimensions_and_type(env)
    hash_function = env.hash_state if hasattr(env, 'hash_state') else None
    return Task(name, observation_dims, observation_type, action_dims, action_type, hash_function)
