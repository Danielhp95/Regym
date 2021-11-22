import typing
from typing import Optional, Dict, List
from functools import reduce

import gym

from .gym_parser import parse_gym_environment
from .unity_parser import parse_unity_environment
from .tasks import Task
from .env_type import EnvType


def generate_task(env_name: str,
                  env_type: EnvType = EnvType.SINGLE_AGENT,
                  wrappers: List[gym.Wrapper] = [],
                  **env_kwargs: Optional[Dict]) -> Task:
    '''
    Returns a regym.environments.Task by creating an environment derived from :param: env_name
    optionally parameterized by :param: env_kwargs. The resulting Task extracts relevant information
    used for build regym.rl_algorithms.agents able to act in said Task.
    If :param: env_name matches a registered OpenAI Gym environment it will create it from there
    If :param: env_name points to a (platform specific) UnityEnvironment executable, it will generate a Unity environment
    In the case of :param: env_name being detected as both an OpenAI Gym and Unity environmet, an error will be raised

    :param env_name: String identifier for the environment to be created
    :param env_type: Determines whether the parameter is (single/multi)-agent
                     and how are the environment processes these actions
                     (i.e all actions simultaneously, or sequentially)
    :param env_wrappers: Wrapper or list of gym Wrappers to modify the environment.
    :param env_kwargs: Keyword arguments to be passed as parameters to the underlying environment.
                   These will be applied first, and wrappers later
    :returns: Task created from :param: env_name
    '''
    if env_name is None: raise ValueError('Parameter \'env_name\' was None')
    is_gym_environment = any([env_name == spec.id for spec in gym.envs.registry.all()]) # Checks if :param: env_name was registered
    is_unity_environment = check_for_unity_executable(env_name)
    if is_gym_environment and is_unity_environment: raise ValueError(f'{env_name} exists as both a Gym and an Unity environment. Rename Unity environment to remove duplicate problem.')
    if is_gym_environment:
        initial_env = gym.make(env_name, **env_kwargs)
        wrapped_environment = reduce(lambda env, wrapper: wrapper(env), wrappers, initial_env)
        return parse_gym_environment(wrapped_environment, env_type, wrappers)
    if is_unity_environment: return parse_unity_environment(env_name, env_type, env_kwargs)
    else: raise ValueError(f'Environment \'{env_name}\' was not recognized as either a Gym nor a Unity environment')


def check_for_unity_executable(env_name):
    '''
    Checks if :param: env_name points to a Unity Executable
    :param env_name: String identifier for the environment to be created
    :returns: Boolean whether :param: env_name is a Unity executable
    '''
    import os, platform
    valid_extensions = {'Linux': '.x86_64', 'Darwin': '.app', 'Windows': '.exe'}
    return os.path.isfile(env_name + valid_extensions[platform.system()])
