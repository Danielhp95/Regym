import itertools
import numpy as np
from gym import spaces
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple

from gym_rock_paper_scissors.envs.one_hot_space import OneHotEncoding

from .task import Task


def parse_gym_environment(env):
    '''
    Assumptions:
        - Observation / Action space (it's geometry, dimensionality) is identical for all agents
    :param env: multiagent environment.
    '''
    name = env.spec.id
    action_dims, action_type = get_action_dimensions_and_type(env)
    observation_dims, observation_type = get_observation_dimensions_and_type(env)
    state_space_size = env.state_space_size if hasattr(env, 'state_space_size') else None
    action_space_size = env.action_space_size if hasattr(env, 'action_space_size') else None
    hash_function = env.hash_state if hasattr(env, 'hash_state') else None
    return Task(name, state_space_size, action_space_size, observation_dims, observation_type, action_dims, action_type, hash_function)


def get_observation_dimensions_and_type(env):
    def parse_dimension_space(space):
        if isinstance(space, OneHotEncoding): return space.size, 'Discrete'
        elif isinstance(space, Discrete): return space.n, 'Discrete'
        elif isinstance(space, Box): return int(np.prod(space.shape)), 'Continuous'
        elif isinstance(space, Tuple): return sum([parse_dimension_space(s)[0] for s in space.spaces]), parse_dimension_space(space.spaces[0])[1]
        raise ValueError('Unknown observation space: {}'.format(space))

    if hasattr(env.observation_space, 'spaces') or len(env.observation_space.spaces) <= 1:
        # Multi agent environment
        return parse_dimension_space(env.observation_space.spaces[0])
    else:
        # Single agent environment
        return parse_dimension_space(env.observation_space)


def get_action_dimensions_and_type(env):
    def parse_dimension_space(space):
        if isinstance(space, Discrete): return space.n, 'Discrete'
        elif isinstance(space, MultiDiscrete): return flatten_multidiscrete_action_space(space.nvec), 'Discrete'
        elif isinstance(space, Box): return space.shape[0], 'Continuous'
        else: raise ValueError('Unknown action space: {}'.format(space))

    if hasattr(env.action_space, 'spaces') or len(env.action_space.spaces) <= 1:
        # Multi agent environment
        return parse_dimension_space(env.action_space.spaces[0])
    else:
        # Single agent environment
        return parse_dimension_space(env.action_space)


def flatten_multidiscrete_action_space(action_space):
    """
    Creates a Dict that maps discrete actions (scalars) to branched actions (lists).
    Each key in the Dict maps to one unique set of branched actions, and each value
    contains the List of branched actions.
    """
    possible_vals = [range(_num) for _num in action_space]
    all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
    # Dict should be faster than List for large action spaces
    action_lookup = {_scalar: _action for (_scalar, _action) in enumerate(all_actions)}
    return len(action_lookup)
