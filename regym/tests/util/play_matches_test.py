from typing import List
import pytest
import numpy as np

from regym.rl_algorithms.agents import build_Deterministic_Agent, MixedStrategyAgent
from regym.rl_algorithms import rockAgent, scissorsAgent
from regym.environments import generate_task
from regym.environments import EnvType
from regym.util.play_matches import play_multiple_matches


@pytest.fixture
def RPS_task():
    import gym_rock_paper_scissors
    return generate_task('RockPaperScissors-v0', EnvType.MULTIAGENT_SIMULTANEOUS_ACTION)


@pytest.fixture
def Kuhn_task():
    import gym_kuhn_poker
    return generate_task('KuhnPoker-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)


def test_can_play_simultaneous_action_environments(RPS_task):
    agent_vector = [rockAgent, scissorsAgent]
    play_matches_given_task_and_agent_vector(RPS_task, agent_vector)


def test_can_play_sequential_action_environments(Kuhn_task):
    agent_vector = [build_Deterministic_Agent(Kuhn_task, {'action': 1}, 'DeterministicAgent-1'),
                    build_Deterministic_Agent(Kuhn_task, {'action': 0}, 'DeterministicAgent-0')]
    play_matches_given_task_and_agent_vector(Kuhn_task, agent_vector)


def play_matches_given_task_and_agent_vector(task, agent_vector: List):
    expected_winrates = [1., 0.]
    number_matches = 10

    winrates, trajectories = play_multiple_matches(task, agent_vector,
                                                   n_matches=number_matches,
                                                   keep_trajectories=True)

    assert len(trajectories) == number_matches
    assert task.total_episodes_run == number_matches
    np.testing.assert_array_equal(expected_winrates, winrates)


def test_play_matches_can_shuffle_agent_positions(RPS_task):
    rockAgent_1 = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent1')
    rockAgent_2 = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent2')
    agent_vector = [rockAgent_1, rockAgent_2]
    expected_winrates = [0.5, 0.5]
    actual_winrates = play_multiple_matches(RPS_task, agent_vector,
                                            n_matches=500,
                                            shuffle_agent_positions=True)
    np.testing.assert_allclose(expected_winrates, actual_winrates,
                               atol=0.10)
