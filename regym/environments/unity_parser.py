from obstacle_tower_env import ObstacleTowerEnv
from .gym_parser import parse_gym_environment


def parse_unity_environment(env_name):
    if env_name != 'obstacletower': raise ValueError('Only obstacletower environment currently supported')
    env = ObstacleTowerEnv(env_name, retro=True, realtime_mode=False)
    return parse_gym_environment(env)
