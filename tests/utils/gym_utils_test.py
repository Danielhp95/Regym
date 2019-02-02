import os
import sys
sys.path.append(os.path.abspath('../../'))
import pytest

import util
import gym


@pytest.fixture
def robosumo_env():
    import robosumo
    return gym.make('RoboSumo-Ant-vs-Ant-v0')


@pytest.fixture
def RPS_env():
    '''
    Assumes that RPS uses a recall of 3
    '''
    import gym_rock_paper_scissors
    return gym.make('RockPaperScissors-v0')


@pytest.fixture
def single_agent_env():
    return gym.make('CartPole-v0')


def test_robosumo_get_observation_dimensions(robosumo_env):
    compare_dimensions(expected=120, actual=util.gym_utils.get_observation_dimensions(robosumo_env))


def test_robosumo_get_action_dimensions(robosumo_env):
    compare_dimensions(expected=8, actual=util.gym_utils.get_action_dimensions(robosumo_env))


def test_RPS_get_observation_dimensions(RPS_env):
    compare_dimensions(expected=6, actual=util.gym_utils.get_observation_dimensions(RPS_env))


def test_RPS_get_action_dimensions(RPS_env):
    compare_dimensions(expected=3, actual=util.gym_utils.get_action_dimensions(RPS_env))


def compare_dimensions(actual, expected):
    assert actual == expected


def test_fail_single_agent_environment(single_agent_env):
    with pytest.raises(ValueError) as _:
        util.gym_utils.get_observation_dimensions(single_agent_env)
