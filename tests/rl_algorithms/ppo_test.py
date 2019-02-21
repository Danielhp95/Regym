import os
import sys
sys.path.append(os.path.abspath('../../'))

from rl_algorithms.PPO import PPOAlgorithm
from environments.gym_parser import parse_gym_environment
import pytest


@pytest.fixture
def RPSTask():
    import gym
    import gym_rock_paper_scissors
    return parse_gym_environment(gym.make('RockPaperScissors-v0'))


@pytest.fixture
def ppo_config_dict():
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = True
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 256
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['episode_interval_per_training'] = 8
    return config


@pytest.fixture
def ppo_kwargs(ppo_config_dict):
    return ppo_config_dict


def test_creation_ppo_algorithm_from_config(RPSTask, ppo_kwargs):
    algorithm = PPOAlgorithm(ppo_kwargs)
    assert all([i in ppo_kwargs.items() for i in algorithm.config.items()])
