from typing import Callable, Union, Optional, List, Tuple
from functools import reduce

import sys
import torch

from multiprocessing import Queue, cpu_count
from multiprocessing.sharedctypes import SynchronizedArray
from multiprocessing.connection import Connection

import gym
from gym.vector.utils import (create_shared_memory, create_empty_array,
                              write_to_shared_memory,
                              concatenate, CloudpickleWrapper, clear_mpi_env_vars)
from gym.vector import AsyncVectorEnv


class RegymAsyncVectorEnv(AsyncVectorEnv):

    def __init__(self, env_name: str, num_envs: int):
        '''
        Extension of OpenAI Gym's AsyncVectorEnv which also supports
        retrieving a copy of each of the underlying environments inside of the
        AsyncVectorEnv (via `get_envs()` method). This extra feature is key for
        model based learning, as such algorithms require a copy of the
        environment at every step.

        :param env_name: Name of OpenAI Gym environment
        :param num_envs: Number of parallel environments to run.
        '''
        if num_envs == -1: num_envs = cpu_count()
        worker = _regym_worker_shared_memory
        env_fns = [self._make_env_fn(env_name) for _ in range(num_envs)]
        super().__init__(env_fns,
                         observation_space=None, action_space=None, # Default params
                         shared_memory=True, copy=True,  # Default parameters
                         context=None, daemon=True,      # Default parameters
                         worker=worker)

    def get_envs(self) -> List[gym.Env]:
        '''
        Copies the environment of all the underlying processes.
        :returns: List of copies of the environments handled by :self: worker
                  (one for each process)
        '''
        for pipe in self.parent_pipes: pipe.send(('environment', None))
        envs, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        return envs

    def _make_env_fn(self, env_name: str) -> Callable[[], gym.Env]:
        '''
        Creates a function that takes no arguments and generates an instance of
        :param: env_name.

        NOTE: Because of `multiprocessing.set_start_method()`,
        if we want to generate a function that uses an environment which by
        default is not in Gym's registry (built-in environments), we need to modify
        this source code of this function to add the import statement to the
        package defining such environment.

        :param env_name: Name of the OpenAI Gym environment
        :returns: Environment generation function
        '''
        def _make_env_from_name():
            # Necessary hack. Import other env names if necessary.
            import gym_connect4
            return gym.make(env_name)
        return _make_env_from_name


def _regym_worker_shared_memory(index: int, env_fn: Callable[[], gym.Env],
                                pipe: Connection, parent_pipe: Connection,
                                shared_memory: Tuple[SynchronizedArray],
                                error_queue: Queue):
    '''
    Based on function `gym.vector.async_vector_env._worker_shared_memory`
    See that function's documentation

    Custom additions:
        - 'environment' command: To return underlying environment
        - 'step' command returns:
            Note: succ_obs dimensions:
            [num_agents, num_environments, environment_observations]
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
