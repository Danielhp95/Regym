import os
import sys
sys.path.append(os.path.abspath('../../'))
import pytest

import environments
from environments.gym_parser import Task
import gym


# @pytest.fixture
# def robosumo_env():
#     import robosumo
#     return gym.make('RoboSumo-Ant-vs-Ant-v0')


@pytest.fixture
def RPS_env():
    '''
    Assumes that RPS uses a recall of 3
    (env param stacked_observations = 3)
    '''
    import gym_rock_paper_scissors
    return gym.make('RockPaperScissors-v0')


@pytest.fixture
def single_agent_env():
    return gym.make('CartPole-v0')


# TODO get Mujoco License for somebody other than Daniel Hernandez
# def test_robosumo_get_observation_dimensions(robosumo_env):
#     expected_observation_dim = 120
#     expected_observation_type = 'Continuous'
#     observation_dims, observation_type = environments.gym_parser.get_observation_dimensions_and_type(robosumo_env)
#     assert expected_observation_dim == observation_dims
#     assert expected_observation_type == observation_type
#
#
# def test_robosumo_get_action_dimensions(robosumo_env):
#     expected_action_dim = 120 # TODO look at shape of (Box) actionspace
#     expected_action_type = 'Continuous'
#     action_dims, action_type = environments.gym_parser.get_action_dimensions_and_type(robosumo_env)
#     assert expected_action_dim == action_dims
#     assert expected_action_type == action_type


def test_RPS_get_observation_dimensions(RPS_env):
    expected_observation_dim = 30
    expected_observation_type = 'Discrete'
    observation_dims, observation_type = environments.gym_parser.get_observation_dimensions_and_type(RPS_env)
    assert expected_observation_dim == observation_dims
    assert expected_observation_type == observation_type


def test_RPS_get_action_dimensions(RPS_env):
    expected_action_dim = 3
    expected_action_type = 'Discrete'
    action_dims, action_type = environments.gym_parser.get_action_dimensions_and_type(RPS_env)
    assert expected_action_dim == action_dims
    assert expected_action_type == action_type


def test_task_creation(RPS_env):
    expected_observation_dim = 30
    expected_observation_type = 'Discrete'
    expected_action_space_size = 3
    expected_action_dim = 3
    expected_action_type = 'Discrete'
    expected_hash_function = RPS_env.hash_state
    expected_state_space_size = RPS_env.state_space_size
    rps_task = Task(RPS_env.spec.id, expected_state_space_size, expected_action_space_size, expected_observation_dim, expected_observation_type, expected_action_dim, expected_action_type, expected_hash_function)
    assert rps_task == environments.parse_gym_environment(RPS_env)


def test_fail_single_agent_environment(single_agent_env):
    with pytest.raises(ValueError) as _:
        environments.gym_parser.get_observation_dimensions_and_type(single_agent_env)
