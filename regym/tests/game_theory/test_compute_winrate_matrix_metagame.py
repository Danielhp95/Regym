import pytest
import numpy as np

from regym.rl_algorithms.agents import rockAgent, paperAgent, scissorsAgent
from regym.game_theory import (compute_winrate_matrix_metagame,
                               generate_evaluation_matrix_multi_population,
                               relative_population_performance,
                               evolution_relative_population_performance)
from regym.rl_algorithms import build_Random_Agent
from regym.rl_algorithms.agents import MixedStrategyAgent, rockAgent, paperAgent, scissorsAgent


from test_fixtures import ppo_config_dict, RPSTask, pendulum_task, Connect4Task


def test_for_compute_winrate_matrix_metagame_none_population_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        _ = compute_winrate_matrix_metagame(population=None,
                                            episodes_per_matchup=10,
                                            task=RPSTask)


def test_for_compute_winrate_matrix_metagame_empty_population_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        _ = compute_winrate_matrix_metagame(population=[],
                                            episodes_per_matchup=10,
                                            task=RPSTask)


def test_for_compute_winrate_matrix_metagame_negative_or_zero_episodes_per_matchup_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        _ = compute_winrate_matrix_metagame(population=[1, 2, 3],
                                            episodes_per_matchup=-1,
                                            task=RPSTask)

    with pytest.raises(ValueError) as _:
        _ = compute_winrate_matrix_metagame(population=[1, 2, 3],
                                            episodes_per_matchup=0,
                                            task=RPSTask)


def test_for_compute_winrate_matrix_metagame_singleagent_task_raises_valueerror(pendulum_task):
    with pytest.raises(ValueError) as _:
        _ = compute_winrate_matrix_metagame(population=[1, 2, 3],
                                            episodes_per_matchup=10,
                                            task=pendulum_task)


def test_for_generate_evaluation_matrix_multi_population_singleagent_task_raises_valueerror(pendulum_task):
    with pytest.raises(ValueError) as _:
        _ = generate_evaluation_matrix_multi_population(task=pendulum_task,
                                                        populations=[[None], [None]],
                                                        episodes_per_matchup=10)

def test_for_generate_evaluation_matrix_multi_population_single_population_raises_valueerror(RPSTask):
    with pytest.raises(ValueError) as _:
        _ = generate_evaluation_matrix_multi_population(task=RPSTask,
                                                        populations=[[None]],
                                                        episodes_per_matchup=10)


def test_single_agent_population(RPSTask):
    population = [rockAgent]
    expected_winrate_matrix = np.array([[0.5]])

    actual_winrate_matrix = compute_winrate_matrix_metagame(population=population,
                                                            episodes_per_matchup=1,
                                                            task=RPSTask,
                                                            num_envs=1)

    np.testing.assert_array_equal(expected_winrate_matrix, actual_winrate_matrix)


def test_can_compute_rock_paper_scissors_metagame(RPSTask):
    population = [rockAgent, paperAgent, scissorsAgent]
    expected_winrate_matrix = np.array([[0.5, 0., 1.],
                                        [1., 0.5, 0.],
                                        [0., 1., 0.5]])

    actual_winrate_matrix = compute_winrate_matrix_metagame(population=population,
                                                            episodes_per_matchup=5,
                                                            task=RPSTask,
                                                            num_envs=1)

    np.testing.assert_array_equal(expected_winrate_matrix, actual_winrate_matrix)


def test_integration_random_agent_rock_paper_scissors(RPSTask):
    population = [build_Random_Agent(RPSTask, {}, 'Test-1'),
                  build_Random_Agent(RPSTask, {}, 'Test-2')]
    winrate_matrix_metagame = compute_winrate_matrix_metagame(population=population,
                                                              episodes_per_matchup=5,
                                                              task=RPSTask,
                                                              num_envs=1)

    # Diagonal winrates are all 0.5
    np.testing.assert_allclose(winrate_matrix_metagame.diagonal(),
                               np.full(winrate_matrix_metagame.diagonal().shape, 0.5))
    # a_i,j + a_j,i = 1 for all non diagonal entries
    for i, j in zip(*np.triu_indices_from(winrate_matrix_metagame, k=1)):
        complementary_sum = winrate_matrix_metagame[i, j] + winrate_matrix_metagame[j, i]
        np.testing.assert_allclose(complementary_sum, 1.)


def test_can_generate_evaluation_matrix_multiple_population(RPSTask):
    population_1 = [rockAgent, rockAgent]
    population_2 = [paperAgent, paperAgent]
    populations=[population_1, population_2]

    expected_evaluation_matrix = np.array([[0., 0.],
                                           [0., 0.]])
    actual_evaluation_matrix = generate_evaluation_matrix_multi_population(
            populations=populations, task=RPSTask, episodes_per_matchup=10)

    np.testing.assert_array_equal(expected_evaluation_matrix, actual_evaluation_matrix)

    # When reversing the populations, we should have an evaluation matrix of all 1s

    populations=[population_2, population_1]

    expected_evaluation_matrix = np.array([[1., 1.],
                                           [1., 1.]])
    actual_evaluation_matrix = generate_evaluation_matrix_multi_population(
            populations=populations, task=RPSTask, episodes_per_matchup=10)

    np.testing.assert_array_equal(expected_evaluation_matrix, actual_evaluation_matrix)


def test_can_compute_relative_population_performance(RPSTask):
    population_1 = [rockAgent, rockAgent]
    population_2 = [paperAgent, paperAgent]

    # rock always loses vs paper. So we expect a winrate of rock vs paper of 0%
    # REMEMBER: relative population performance is computed over evaluation matrices
    # which are antisymmetric around 0. So min=-0.5, max=0.5.
    expected_relative_population_performance = -0.5

    actual_relative_pop_performance = relative_population_performance(
                 population_1=population_1, population_2=population_2,
                 task=RPSTask, episodes_per_matchup=10)

    np.testing.assert_allclose(actual_relative_pop_performance, expected_relative_population_performance)

    # When reversing the populations, we should have a relative_population_performance of 0.5

    expected_relative_population_performance = 0.5

    actual_relative_pop_performance = relative_population_performance(
                 population_1=population_2, population_2=population_1,
                 task=RPSTask, episodes_per_matchup=10)

    np.testing.assert_allclose(actual_relative_pop_performance, expected_relative_population_performance)


def test_can_compute_evolution_of_relative_population_performance(RPSTask):
    rockAgent_1 = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent1')
    rockAgent_2 = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent2')
    rockAgent_3 = MixedStrategyAgent(support_vector=[1, 0, 0], name='RockAgent3')

    population_1 = [rockAgent_1, paperAgent]
    population_2 = [rockAgent_2, rockAgent_3]

    expected_evolution_relative_population_performance = [0, 0.5]

    actual_evolution_rel_pop_perf = evolution_relative_population_performance(
            population_1=population_1, population_2=population_2,
            task=RPSTask, episodes_per_matchup=500)
    np.testing.assert_allclose(expected_evolution_relative_population_performance,
                               actual_evolution_rel_pop_perf, atol=0.05)


def test_compute_evolution_of_relative_population_performance_is_faster_in_parallel(Connect4Task):
    import torch
    from time import time
    from regym.networks.preprocessing import batch_vector_observation
    from copy import deepcopy

    agent_1 = torch.load('./1024_iterations.pt')
    agent_2 = torch.load('./11264_iterations.pt')

    population_1 = [agent_1, deepcopy(agent_1), deepcopy(agent_1)]
    population_2 = [agent_2, deepcopy(agent_2), deepcopy(agent_2)]

    start_time_1 = time()
    evolution_relative_population_performance(
        population_1=population_1, population_2=population_2,
        task=Connect4Task, episodes_per_matchup=500,
        num_envs=1)
    end_time_1 = time() - start_time_1

    print('Sequential matches time', end_time_1)

    for agent in population_1:
        agent.state_preprocessing = batch_vector_observation
    for agent in population_2:
        agent.state_preprocessing = batch_vector_observation

    start_time_2 = time()
    evolution_relative_population_performance(
        population_1=population_1, population_2=population_2,
        task=Connect4Task, episodes_per_matchup=40,
        num_envs=-1)
    end_time_2 = time() - start_time_2

    print('Concurrent matches time', end_time_2)
    print('Speedup factor', end_time_1 / end_time_2)
