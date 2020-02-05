import pytest

from regym.environments import generate_task


def test_can_pass_kwargs_to_env():
    from gym.envs.registration import register
    register(id='DummyEnv-v0', entry_point='regym.tests.environments.params_test_env:ParamsTestEnv')

    params = {'param1': 1, 'param2': 2, 'param3': 3}

    task = generate_task('DummyEnv-v0', **params)

    assert task.env.param1 == 1
    assert task.env.param2 == 2
    assert task.env.param3 == 3
