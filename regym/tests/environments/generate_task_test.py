import pytest

from regym.environments import generate_task, EnvType


def test_can_pass_kwargs_to_env():
    from gym.envs.registration import register
    register(id='DummyEnv-v0', entry_point='regym.tests.environments.params_test_env:ParamsTestEnv')

    params = {'param1': 1, 'param2': 2, 'param3': 3}

    task = generate_task('DummyEnv-v0', **params)

    assert task.env.param1 == 1
    assert task.env.param2 == 2
    assert task.env.param3 == 3


def test_can_parse_connect4_task():
    import gym_connect4

    task = generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)

    expected_observation_dim = (3, 7, 6)
    expected_observation_size = 126
    expected_observation_type = 'Continuous'

    assert expected_observation_dim == task.observation_dim
    assert expected_observation_type == task.observation_type
    assert expected_observation_size == task.observation_size
