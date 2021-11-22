import pytest

from test_fixtures import RPSTask, KuhnTask, PendulumTask, CartPoleTask, Connect4Task

import regym
from regym.rl_algorithms import build_Deterministic_Agent, build_Random_Agent
from regym.rl_algorithms.agents import rockAgent


def test_deterministic_agent_raises_exception_on_continuous_action_spaces(PendulumTask):
    with pytest.raises(ValueError) as _:
        _ = build_Deterministic_Agent(PendulumTask, {'action': 0}, 'DeterministicTest')


def test_deterministic_agent_can_act_on_single_agent_env(CartPoleTask):
    expected_action = 0
    agent = build_Deterministic_Agent(
        CartPoleTask,
        {'action': expected_action}, 'DeterministicTestTest')
    trajectory = CartPoleTask.run_episode([agent], training=False)

    assert all(map(lambda a: a == expected_action,
                   extract_actions_from_trajectory(trajectory)))

    expected_action_sequence = [0, 1]
    agent = build_Deterministic_Agent(
        CartPoleTask,
        {'action_sequence': expected_action_sequence}, 'DeterministicTest')

    trajectory = CartPoleTask.run_episode([agent], training=False)

    assert all(map(lambda a: a in expected_action_sequence,
                   extract_actions_from_trajectory(trajectory)))


def test_random_agent_can_act_on_single_agent_env(CartPoleTask):
    action_space = CartPoleTask.env.action_space

    agent = build_Random_Agent(CartPoleTask, {}, 'RandomTest')
    trajectory = CartPoleTask.run_episode([agent], training=False)
    assert all(map(lambda a: action_space.contains(a),
                   extract_actions_from_trajectory(trajectory)))


def test_deterministic_agent_can_act_on_multiagent_sequential_environment(Connect4Task):
    expected_actions = [0, 1]
    agent_1 = build_Deterministic_Agent(
        Connect4Task,
        {'action': expected_actions[0]}, 'DeterministicTest-1')
    agent_2 = build_Deterministic_Agent(
        Connect4Task,
        {'action': expected_actions[1]}, 'DeterministicTest-2')

    trajectory = Connect4Task.run_episode([agent_1, agent_2], training=False)

    for i, (s, a, r, succ_s, o) in enumerate(trajectory):
        assert a == expected_actions[i % 2]


def test_deterministic_agent_can_act_on_async_single_agent(Connect4Task):
    expected_actions = [0, 1]
    agent_1 = build_Deterministic_Agent(
        Connect4Task,
        {'action': expected_actions[0]}, 'DeterministicTest-1')
    agent_2 = build_Deterministic_Agent(
        Connect4Task,
        {'action': expected_actions[1]}, 'DeterministicTest-2')

    trajectories = Connect4Task.run_episodes([agent_1, agent_2], training=False,
                                             num_envs=2, num_episodes=2)

    for trajectory in trajectories:
        for i, (s, a, r, succ_s, o) in enumerate(trajectory):
            assert a == expected_actions[i % 2]


def extract_actions_from_trajectory(trajectory):
    return [timestep[1] for timestep in trajectory]
