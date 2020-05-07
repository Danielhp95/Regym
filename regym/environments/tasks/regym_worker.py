from typing import Callable, Union, Optional, List, Tuple
from functools import reduce

import sys
from multiprocessing import Pipe, Queue
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection

import gym
from gym.vector.utils import (create_shared_memory, create_empty_array,
                              write_to_shared_memory,
                              concatenate, CloudpickleWrapper, clear_mpi_env_vars)
from gym.vector import AsyncVectorEnv


class RegymAsyncVectorEnv(AsyncVectorEnv):

    def __init__(self, env_name, num_envs: int,
                 observation_space: Optional[gym.Space] = None,
                 action_space: Optional[gym.Space] = None,
                 shared_memory=True, copy=True, context=None, daemon=True):
        '''
        TODO
        '''
        worker = _regym_worker_shared_memory
        env_fns = [self._make_env(env_name) for _ in range(num_envs)]
        super().__init__(env_fns, observation_space, action_space,
                         shared_memory, copy, context, daemon, worker)

    def get_envs(self) -> Tuple[gym.Env]:
        '''
        TODO
        '''
        for pipe in self.parent_pipes: pipe.send(('environment', None))
        envs, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        return envs

    def _make_env(self, env_name: str) -> Callable[[], gym.Env]:
        def _make():
            return gym.make(env_name)
        return _make


def _regym_worker_shared_memory(index: int, env_fn: Callable[[], gym.Env],
                                pipe: Connection, parent_pipe: Connection,
                                shared_memory: Tuple[SynchronizedArray],
                                error_queue: Queue):
    '''
    Based on function `gym.vector.async_vector_env._worker_shared_memory`

    Custom additions:
        - 'environment' command: To return underlying environment
        - 'step' command returns : ???
    '''
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send((None, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send(((None, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == 'environment':
                pipe.send((env, True))
            elif command == '_check_observation_space':
                pipe.send((data == observation_space, True))
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, '
                    '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
