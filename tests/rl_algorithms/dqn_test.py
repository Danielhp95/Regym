import os
import sys
sys.path.append(os.path.abspath('../../'))

from rl_algorithms.TQL import TabularQLearningAlgorithm
from rl_algorithms.agents import build_DQN_Agent
from environments.gym_parser import parse_gym_environment
import pytest


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
    config['use_cuda'] = False
    config['use_PER'] = False
    config['PER_alpha'] = 0.07
    config['min_memory'] = 5.0e1
    config['memoryCapacity'] = 25.0e3
    config['nbrTrainIteration'] = 32
    return config


# TODO
# def test_build_dqn(RPSTask, dqn_config_dict):
#     training = False
#     agent = build_DQN_Agent(RPSTask, dqn_config_dict)
#     assert agent.algorithm.Q_table.shape == (RPSTask.state_space_size, RPSTask.action_space_size)
#     assert agent.algorithm.learning_rate == dqn_config_dict['learning_rate']
#     assert agent.algorithm.hashing_function == RPSTask.hash_function
#     assert agent.algorithm.training == training


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


#def learns_to_beat_rock_in_RPS(RPSenv, RPSTask, dqn_config_dict):
#    '''
#    Test used to make sure that agent is 'learning' by learning a best response
#    against an agent that only plays rock in rock paper scissors.
#    i.e from random, learns to play only (or mostly) paper
#    '''
#    agent = build_DQN_Agent(RPSTask, dqn_config_dict)


# def can_be_used_with_agent_hook_test()
