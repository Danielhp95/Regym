import os
import sys
sys.path.append(os.path.abspath('../../'))
import pytest

import util


@pytest.fixture
def env():
    import gym
    import robosumo
    return gym.make('RoboSumo-Ant-vs-Ant-v0')


@pytest.fixture
def single_agent_env():
    import gym
    return gym.make('CartPole-v0')


def test_get_observation_dimensions(env):
    expected_space_dimensions = 120
    space_dimensions = util.gym_utils.get_observation_dimensions(env)
    assert space_dimensions == expected_space_dimensions


def test_get_action_dimensions(env):
    expected_space_dimensions = 120
    space_dimensions = util.gym_utils.get_observation_dimensions(env)
    assert space_dimensions == expected_space_dimensions


def test_fail_single_agent_environment(single_agent_env):
    with pytest.raises(ValueError) as _:
        util.gym_utils.get_observation_dimensions(single_agent_env)
