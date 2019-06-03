import gym

from .gym_parser import parse_gym_environment
from .unity_parser import parse_unity_environment


def parse_environment(env_name):
    is_gym_environment = any([env_name == spec.id for spec in gym.envs.registry.all()]) # Checks if :param: env_name was registered
    is_unity_environment = check_for_unity_executable(env_name)
    if is_gym_environment and is_unity_environment: raise ValueError(f'{env_name} exists as both a Gym and an Unity environment. Rename Unity environment to remove duplicate problem.')
    if is_gym_environment: return parse_gym_environment(gym.make(env_name))
    if is_unity_environment: return parse_unity_environment(env_name)
    else: raise ValueError('Environment \'{env_name}\' was not recognized as either a Gym nor a Unity environment')


def check_for_unity_executable(env_name):
    import os, platform
    valid_extensions = {'Linux': '.x86_64', 'Darwin': '.app', 'Windows': '.exe'}
    return os.path.isfile(env_name + valid_extensions[platform.system()])
