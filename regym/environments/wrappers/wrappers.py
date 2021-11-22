from collections import deque
import numpy as np

import gym
from gym.spaces import Box
from gym import Wrapper
from gym.wrappers import LazyFrames


class FrameStack(Wrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    .. note::

        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.

    .. note::

        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.

    Example::

        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)

    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally

    """
    def __init__(self, env, num_stack, lz4_compress=False):
        super(FrameStack, self).__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        self.is_env_multiagent = isinstance(self.observation_space, gym.spaces.Tuple)

        # Maybe there's a better way of figuring out if we are on
        # a multiagent environment? Like passing a param to this wrapper
        if self.is_env_multiagent:
            # We are on a multigent environment
            num_players = len(self.observation_space)
            # ASSUMPTION, all players share observation space
            self.single_player_observation_space = self.observation_space[0]
            low = np.repeat(self.single_player_observation_space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(self.single_player_observation_space.high[np.newaxis, ...], num_stack, axis=0)

            stacked_single_player_observation_space = Box(
                low=low, high=high,
                dtype=self.single_player_observation_space.dtype)
            # ASSUMPTION, all players share observation space
            self.observation_space = gym.spaces.Tuple(
                [stacked_single_player_observation_space for _ in range(num_players)])

        else:  # Single agent environment
            low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
            high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)
            self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def _get_observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        observation = np.array(self.frames)
        if self.is_env_multiagent:
            observation = self._regym_multiagent_reshape(observation)
        return observation

    def _regym_multiagent_reshape(self, observation: np.ndarray) -> np.ndarray:
        '''
        Without this function, the shape of an observation for a multiagent
        environment would be:
            (num_stacks, num_agents, *observation.shape)  [This is unwanted]
        With this reshaping we turn it into what regym.rl_loops.multiagent_loops
        expect, which is:
            (num_agents, num_stack, *observation.shape)
        '''
        assert observation.shape[0] == self.num_stack, ('First dimension of observation '
                                                        'should be the same as the '
                                                        f'number of stack frames: '
                                                        f'Num stack: {self.num_stack} '
                                                        f'Given: {observation.shape[0]}')
        num_agents = observation.shape[1]
        target_shape = (num_agents, self.num_stack, *self.single_player_observation_space.shape)
        reshaped_observation = np.reshape(observation, target_shape)
        return reshaped_observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self._get_observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self._get_observation()
