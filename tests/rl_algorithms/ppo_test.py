import os
import sys
sys.path.append(os.path.abspath('../../'))
import pytest

import torch

from rl_algorithms.agents import build_PPO_Agent
from rl_algorithms.PPO import PPOAlgorithm
from rl_algorithms import rockAgent
from environments.gym_parser import parse_gym_environment
from multiagent_loops import simultaneous_action_rl_loop

from unittest.mock import Mock


@pytest.fixture
def RPSenv():
    import gym
    import gym_rock_paper_scissors
    return gym.make('RockPaperScissors-v0')


@pytest.fixture
def RPSTask(RPSenv):
    return parse_gym_environment(RPSenv)


@pytest.fixture
def ppo_config_dict():
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = False
    config['use_cuda'] = False
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 256
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['horizon'] = 1024
    return config


@pytest.fixture
def ppo_kwargs(ppo_config_dict):
    kwargs = ppo_config_dict.copy()
    mockNN = Mock(torch.nn.Module)
    mockNN.parameters.return_value = []
    kwargs['model'] = mockNN
    return kwargs


# def test_creation_ppo_algorithm_from_kwargs(RPSTask, ppo_kwargs):
#     algorithm = PPOAlgorithm(ppo_kwargs)
#     print(ppo_kwargs.items())
#     for kwarg in algorithm.kwargs.items():
#         print(kwarg)
#         assert kwarg in ppo_kwargs.items()


def test_ppo_can_take_actions(RPSenv, RPSTask, ppo_config_dict):
    agent = build_PPO_Agent(RPSTask, ppo_config_dict)
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = RPSenv.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        observation, rewards, done, info = RPSenv.step([a, a])
        # TODO technical debt
        # assert RPSenv.observation_space.contains([a, a])
        # assert RPSenv.action_space.contains([a, a])


def test_learns_to_beat_rock_in_RPS(RPSTask, ppo_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from rps_test import learns_against_fixed_opponent_RPS

    agent = build_PPO_Agent(RPSTask, ppo_config_dict)
    assert agent.training
    learns_against_fixed_opponent_RPS(agent, fixed_opponent=rockAgent,
                                      training_episodes=1000, inference_percentage=0.9,
                                      reward_threshold=0.1)

# def test_creation_ppo_agent_from_config(RPSTask, ppo_config_dict): #     assert False
