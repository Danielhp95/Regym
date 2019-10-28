import pytest
import unittest
import numpy as np

from regym.game_theory import compute_winrate_matrix_metagame
from regym.rl_algorithms import build_Reinforce_Agent, build_PPO_Agent

from test_fixtures import ppo_config_dict, RPSTask


def test_for_none_population_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        _ = compute_winrate_matrix_metagame(population=None,
                                            episodes_per_matchup=10,
                                            env=RPSTask.env)

def test_for_empty_population_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        _ = compute_winrate_matrix_metagame(population=[],
                                            episodes_per_matchup=10,
                                            env=RPSTask.env)

def test_for_negative_or_zero_episodes_per_matchup_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        # TODO: is it worth it to pass an agent?
        _ = compute_winrate_matrix_metagame(population=[1,2,3],
                                            episodes_per_matchup=-1,
                                            env=RPSTask.env)

    with pytest.raises(ValueError) as _:
        # TODO: is it worth it to pass an agent?
        _ = compute_winrate_matrix_metagame(population=[1,2,3],
                                            episodes_per_matchup=0,
                                            env=RPSTask.env)

def test_for_none_environment_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        # TODO: is it worth it to pass an agent?
        _ = compute_winrate_matrix_metagame(population=[1,2,3],
                                            episodes_per_matchup=10,
                                            env=None)

# TODO: add test to check if environment is multi_agent or single_agent?

def test(ppo_config_dict, RPSTask):
    population = [build_PPO_Agent(RPSTask, ppo_config_dict, 'Test-1'),
                  build_PPO_Agent(RPSTask, ppo_config_dict.copy(), 'Test-2')]
    winrate_matrix_metagame = compute_winrate_matrix_metagame(population=population,
                                                              episodes_per_matchup=5,
                                                              env=RPSTask.env,
                                                              num_workers=1)

    # Diagonal winrates are all 0.5
    np.testing.assert_allclose(winrate_matrix_metagame.diagonal(),
                               np.full(winrate_matrix_metagame.diagonal().shape, 0.5))
    # a_i,j + a_j,i = 1 for all non diagonal entries
    for i, j in zip(*np.triu_indices_from(winrate_matrix_metagame, k=1)):
        complementary_sum = winrate_matrix_metagame[i, j] + winrate_matrix_metagame[j, i]
        np.testing.assert_allclose(complementary_sum, 1.)
