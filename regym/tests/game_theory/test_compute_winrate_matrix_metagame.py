import pytest
import numpy as np

from regym.rl_algorithms.agents import rockAgent, paperAgent, scissorsAgent
from regym.game_theory import compute_winrate_matrix_metagame
from regym.rl_algorithms import build_Reinforce_Agent, build_PPO_Agent

from test_fixtures import ppo_config_dict, RPSTask, pendulum_task


def test_for_none_population_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        _ = compute_winrate_matrix_metagame(population=None,
                                            episodes_per_matchup=10,
                                            task=RPSTask)


def test_for_empty_population_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        _ = compute_winrate_matrix_metagame(population=[],
                                            episodes_per_matchup=10,
                                            task=RPSTask)


def test_for_negative_or_zero_episodes_per_matchup_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        # TODO: is it worth it to pass an agent?
        _ = compute_winrate_matrix_metagame(population=[1, 2, 3],
                                            episodes_per_matchup=-1,
                                            task=RPSTask)

    with pytest.raises(ValueError) as _:
        # TODO: is it worth it to pass an agent?
        _ = compute_winrate_matrix_metagame(population=[1, 2, 3],
                                            episodes_per_matchup=0,
                                            task=RPSTask)


def test_for_singleagent_task_raises_valueerror(pendulum_task):
    with pytest.raises(ValueError) as _:
        # TODO: is it worth it to pass an agent?
        _ = compute_winrate_matrix_metagame(population=[1, 2, 3],
                                            episodes_per_matchup=10,
                                            task=pendulum_task)


def test_single_agent_population(RPSTask):
    population = [rockAgent]
    expected_winrate_matrix = np.array([[0.5]])

    actual_winrate_matrix = compute_winrate_matrix_metagame(population=population,
                                                            episodes_per_matchup=1,
                                                            task=RPSTask,
                                                            num_workers=1)

    np.testing.assert_array_equal(expected_winrate_matrix, actual_winrate_matrix)


def test_can_compute_rock_paper_scissors_metagame(RPSTask):
    population = [rockAgent, paperAgent, scissorsAgent]
    expected_winrate_matrix = np.array([[0.5, 0., 1.],
                                        [1., 0.5, 0.],
                                        [0., 1., 0.5]])

    actual_winrate_matrix = compute_winrate_matrix_metagame(population=population,
                                                            episodes_per_matchup=5,
                                                            task=RPSTask,
                                                            num_workers=1)

    np.testing.assert_array_equal(expected_winrate_matrix, actual_winrate_matrix)


def test_integration_ppo_rock_paper_scissors(ppo_config_dict, RPSTask):
    population = [build_PPO_Agent(RPSTask, ppo_config_dict, 'Test-1'),
                  build_PPO_Agent(RPSTask, ppo_config_dict.copy(), 'Test-2')]
    winrate_matrix_metagame = compute_winrate_matrix_metagame(population=population,
                                                              episodes_per_matchup=5,
                                                              task=RPSTask,
                                                              num_workers=1)

    # Diagonal winrates are all 0.5
    np.testing.assert_allclose(winrate_matrix_metagame.diagonal(),
                               np.full(winrate_matrix_metagame.diagonal().shape, 0.5))
    # a_i,j + a_j,i = 1 for all non diagonal entries
    for i, j in zip(*np.triu_indices_from(winrate_matrix_metagame, k=1)):
        complementary_sum = winrate_matrix_metagame[i, j] + winrate_matrix_metagame[j, i]
        np.testing.assert_allclose(complementary_sum, 1.)
