import os
import sys
sys.path.append(os.path.abspath('../../'))
import pytest

from rl_algorithms.agents import build_DQN_Agent
from rl_algorithms import rockAgent

from environments.gym_parser import parse_gym_environment
from multiagent_loops import simultaneous_action_rl_loop


@pytest.fixture
def RPSenv():
    import gym
    import gym_rock_paper_scissors
    return gym.make('RockPaperScissors-v0')


@pytest.fixture
def RPSTask(RPSenv):
    return parse_gym_environment(RPSenv)


@pytest.fixture
def dqn_config_dict():
    config = dict()
    config['learning_rate'] = 1.0e-3
    config['epsstart'] = 0.8
    config['epsend'] = 0.05
    config['epsdecay'] = 1.0e3
    config['double'] = False
    config['dueling'] = False
    config['use_cuda'] = True
    config['use_PER'] = True
    config['PER_alpha'] = 0.07
    config['min_memory'] = 25.0e1
    config['memoryCapacity'] = 25.0e2
    config['nbrTrainIteration'] = 8
    config['batch_size'] = 256
    config['gamma'] = 0.99
    config['tau'] = 1.0e-3
    return config


# TODO
# def test_build_dqn(RPSTask, dqn_config_dict):
#     agent = build_DQN_Agent(RPSTask, dqn_config_dict)
#     assert False


def test_dqn_can_take_actions(RPSenv, RPSTask, dqn_config_dict):
    agent = build_DQN_Agent(RPSTask, dqn_config_dict)
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = RPSenv.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        observation, rewards, done, info = RPSenv.step([a, a])
        # TODO technical debt
        # assert RPSenv.observation_space.contains([a, a])
        # assert RPSenv.action_space.contains([a, a])


def test_learns_to_beat_rock_in_RPS(RPSenv, RPSTask, dqn_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from rps_test import learns_against_fixed_opponent_RPS

    agent = build_DQN_Agent(RPSTask, dqn_config_dict)
    agent.training = True
    learns_against_fixed_opponent_RPS(agent, fixed_opponent=rockAgent,
                                      training_episodes=200, inference_percentage=0.95,
                                      reward_threshold=0.2)
    