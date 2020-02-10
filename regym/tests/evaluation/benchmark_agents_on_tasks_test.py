from copy import deepcopy
import pytest
import numpy as np

from test_fixtures import RPSTask, RPSTask2Repetitions, pendulum_task, ppo_config_dict
from regym.environments import generate_task, EnvType
from regym.rl_algorithms import build_PPO_Agent
from regym.rl_algorithms import rockAgent, paperAgent, scissorsAgent

from regym.evaluation import benchmark_agents_on_tasks


def test_zero_or_negative_episodes_raises_value_exception(RPSTask, ppo_config_dict):
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'Test-PPO')
    with pytest.raises(ValueError) as _:
        _ = benchmark_agents_on_tasks(tasks=[RPSTask],
                                      agents=[agent],
                                      num_episodes=-1)
    with pytest.raises(ValueError) as _:
        _ = benchmark_agents_on_tasks(tasks=[RPSTask],
                                      agents=[agent],
                                      num_episodes=0)


def test_all_tasks_must_be_single_agent_or_multiagent(RPSTask, pendulum_task, ppo_config_dict):
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'Test-PPO')
    with pytest.raises(ValueError) as _:
        _ = benchmark_agents_on_tasks(tasks=[RPSTask, pendulum_task],
                                      agents=[agent],
                                      num_episodes=1)


def test_if_populate_all_agents_is_not_set_having_fewer_or_more_agents_raises_value_error(RPSTask, ppo_config_dict):
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'Test-PPO')
    with pytest.raises(ValueError) as _:
        _ = benchmark_agents_on_tasks(tasks=[RPSTask],
                                      agents=[agent],
                                      num_episodes=1,
                                      populate_all_agents=False)
    with pytest.raises(ValueError) as _:
        _ = benchmark_agents_on_tasks(tasks=[RPSTask],
                                      agents=[agent, agent, agent],
                                      num_episodes=1,
                                      populate_all_agents=False)

def test_if_populate_all_agents_is_set_a_single_agent_must_be_provided(RPSTask, ppo_config_dict):
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'Test-PPO')
    with pytest.raises(ValueError) as _:
        _ = benchmark_agents_on_tasks(tasks=[RPSTask],
                                      agents=[agent, agent],
                                      num_episodes=1,
                                      populate_all_agents=True)


def test_single_agent_tasks_only_accept_one_agent(pendulum_task, ppo_config_dict):
    agent = build_PPO_Agent(pendulum_task, ppo_config_dict, 'Test-PPO')
    with pytest.raises(NotImplementedError) as _:
        _ = benchmark_agents_on_tasks(tasks=[pendulum_task],
                                      agents=[agent, agent],
                                      num_episodes=1)


def test_can_compute_winrate_for_player1_multiagent_task(RPSTask):
    expected_winrates = [0, 1]

    vs_paper = deepcopy(RPSTask)
    vs_scissors = deepcopy(RPSTask)

    # Ugly, would be awesome to have it in a one line
    vs_paper.extend_task(agents={1: paperAgent})
    vs_scissors.extend_task(agents={1: scissorsAgent})

    actual_winrates = benchmark_agents_on_tasks(tasks=[vs_paper, vs_scissors],
                                                agents=[rockAgent],
                                                num_episodes=10)
    np.testing.assert_array_equal(expected_winrates, actual_winrates)


def test_can_compute_average_reward_for_agent_single_agent_task(RPSTask2Repetitions):
    vs_paper = deepcopy(RPSTask2Repetitions)
    vs_scissors = deepcopy(RPSTask2Repetitions)
    vs_paper.extend_task(agents={1: paperAgent})
    vs_scissors.extend_task(agents={1: scissorsAgent})

    expected_average_rewards = [-1., 1]

    actual_winrates, actual_average_rewards = benchmark_agents_on_tasks(tasks=[vs_paper, vs_scissors],
                                                                        agents=[rockAgent],
                                                                        keep_average_rewards=True,
                                                                        num_episodes=10)
    np.testing.assert_array_equal(expected_average_rewards, actual_average_rewards)


def test_single_agent_can_populate_all_agents(RPSTask):
    expected_winrates = [0.5]

    actual_winrates = benchmark_agents_on_tasks(tasks=[RPSTask],
                                                agents=[rockAgent],
                                                num_episodes=200,
                                                populate_all_agents=True)
    np.testing.assert_allclose(expected_winrates, actual_winrates, atol=0.1)
