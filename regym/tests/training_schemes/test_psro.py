from unittest import mock
import pytest
import numpy as np

from regym.training_schemes import PSRONashResponse
from regym.rl_algorithms import rockAgent, scissorsAgent, paperAgent


def test_task_can_be_generated_from_parameter_env():
    with pytest.raises(ValueError) as _:
        _ = PSRONashResponse(env_id=None,
                             threshold_best_response=0.5,
                             benchmarking_episodes=1)
    with pytest.raises(ValueError) as _:
        _ = PSRONashResponse(env_id='caca',
                             threshold_best_response=0.5,
                             benchmarking_episodes=1)


def test_for_threshold_best_response_must_lay_between_0_and_1():
    with pytest.raises(ValueError) as _:
        _ = PSRONashResponse(env_id='RockPaperScissors-v0',
                             threshold_best_response=-1,
                             benchmarking_episodes=1)
    with pytest.raises(ValueError) as _:
        _ = PSRONashResponse(env_id='RockPaperScissors-v0',
                             threshold_best_response=2,
                             benchmarking_episodes=1)


def test_for_benchmarking_episodes_must_be_strictly_positive():
    with pytest.raises(ValueError) as _:
        _ = PSRONashResponse(env_id='RockPaperScissors-v0',
                             threshold_best_response=0.5,
                             benchmarking_episodes=0)


def test_for_rolling_window_size_must_be_strictly_positive():
    with pytest.raises(ValueError) as _:
        _ = PSRONashResponse(env_id='RockPaperScissors-v0',
                             threshold_best_response=0.5,
                             match_outcome_rolling_window_size=0)


def test_can_fill_missing_game_entries_upon_adding_new_policy():
    psro = PSRONashResponse(env_id='RockPaperScissors-v0', benchmarking_episodes=2)
    psro.menagerie = [rockAgent, paperAgent, scissorsAgent]
    meta_game = np.array([[0.5, 0, np.nan],
                          [1, 0.5, np.nan],
                          [np.nan, np.nan, np.nan]])
    expected_updated_metagame = np.array([[0.5, 0, 1],
                                          [1, 0.5, 0],
                                          [0, 1, 0.5]])
    actual_updated_metagame = psro.fill_meta_game_missing_entries(policies=psro.menagerie,
                                                                  updated_meta_game=meta_game,
                                                                  benchmarking_episodes=psro.benchmarking_episodes,
                                                                  env_id=psro.env_id)
    np.testing.assert_array_equal(expected_updated_metagame, actual_updated_metagame)


def test_can_update_mata_game():
    psro = PSRONashResponse(env_id='RockPaperScissors-v0', benchmarking_episodes=2)
    psro.menagerie = [rockAgent, paperAgent, scissorsAgent]
    psro.meta_game = np.array([[0.5, 0],
                               [1, 0.5]])

    expected_meta_game = np.array([[0.5, 0, 1],
                                   [1, 0.5, 0],
                                   [0, 1, 0.5]])

    actual_meta_game = psro.update_meta_game()

    np.testing.assert_array_equal(expected_meta_game, actual_meta_game)


def test_can_keep_track_of_window_of_winrate_for_learning_policy():
    psro = PSRONashResponse(env_id='RockPaperScissors-v0',
                            match_outcome_rolling_window_size=3)
    training_agent_indeces = [1, 1, 0, 1]
    expected_rolling_window = [1, 0, 1]

    # TODO this is very ugly. It always chooses player 2 (1-index) as winner
    # We should really find a way of mocking this.
    sample_trajectory = [([], [], [0, 1], [])] # (s, a, r, s')
    for i in training_agent_indeces:
        psro.update_rolling_winrates(episode_trajectory=sample_trajectory,
                                     training_agent_index=i)

    np.testing.assert_array_equal(expected_rolling_window,
                                  psro.match_outcome_rolling_window)
