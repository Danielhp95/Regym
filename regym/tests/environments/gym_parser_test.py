import pytest
import gym
from gym.spaces import MultiDiscrete

from regym.environments.task import Task
from regym.environments import gym_parser


@pytest.fixture
def RPS_env():
    '''
    Assumes that RPS uses a recall of 3
    (env param stacked_observations = 3)
    '''
    import gym_rock_paper_scissors
    return gym.make('RockPaperScissors-v0')


def test_multidiscrete_action_flattening():
    space = MultiDiscrete([3, 3, 2, 3])
    expected_action_space_size = 54
    action_space_size = gym_parser.compute_multidiscrete_space_size(space.nvec)
    assert action_space_size == expected_action_space_size


def test_RPS_get_observation_dimensions(RPS_env):
    expected_observation_dim = 30
    expected_observation_type = 'Discrete'
    observation_dims, observation_type = gym_parser.get_observation_dimensions_and_type(RPS_env)
    assert expected_observation_dim == observation_dims
    assert expected_observation_type == observation_type


def test_RPS_get_action_dimensions(RPS_env):
    expected_action_dim = 3
    expected_action_type = 'Discrete'
    action_dims, action_type = gym_parser.get_action_dimensions_and_type(RPS_env)
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
    rps_task = Task(RPS_env.spec.id, RPS_env, expected_state_space_size, expected_action_space_size, expected_observation_dim, expected_observation_type, expected_action_dim, expected_action_type, expected_hash_function)
    assert rps_task == gym_parser.parse_gym_environment(RPS_env)
