import pytest
import numpy as np

from test_fixtures import mcts_config_dict, Connect4Task, RandomWalkTask

from regym.rl_algorithms.agents import build_MCTS_Agent, build_Random_Agent
from regym.util.play_matches import extract_winner



def test_non_integer_budget_raises_value_error(Connect4Task):
    config = {'budget': 0.5}
    with pytest.raises(ValueError) as _:
        _ = build_MCTS_Agent(Connect4Task, config, 'name')


def test_mcts_can_take_actions_discrete_obvservation_discrete_action(Connect4Task, mcts_config_dict):
    mcts1 = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name='MCTS1-test')
    mcts2 = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name='MCTS2-test')
    Connect4Task.run_episode([mcts1, mcts2], training=False)

def test_can_defeat_random_play_in_connect4_both_positions_using_ucb1(Connect4Task, mcts_config_dict):
    mcts_config_dict['selection_phase'] = 'ucb1'
    mcts_config_dict['budget'] = 50
    win_task_in_both_positions(Connect4Task, mcts_config_dict) # First player (index 0) has a much higher budget


def test_can_defeat_random_play_in_connect4_both_positions_using_puct(Connect4Task, mcts_config_dict):
    mcts_config_dict['selection_phase'] = 'puct'
    mcts_config_dict['budget'] = 50
    win_task_in_both_positions(Connect4Task, mcts_config_dict)  # First player (index 0) has a much higher budget


def win_task_in_both_positions(task, mcts_config_dict):
    mcts = build_MCTS_Agent(task, mcts_config_dict, agent_name='MCTS1-test')
    random = build_Random_Agent(task, {}, agent_name='Random-test')
    t1 = task.run_episode([mcts, random], training=False)

    assert extract_winner(t1) == 0  # First player (index 0) has a much higher budget
    t2 = task.run_episode([random, mcts], training=False)
    assert extract_winner(t2) == 1  # Second_player (index 1) has a much higher budget


def test_can_coordinate_in_simulatenous_random_walk(RandomWalkTask, mcts_config_dict):
    mcts_config_dict['budget'] = 50
    mcts_config_dict['rollout_budget'] = 0

    mcts1 = build_MCTS_Agent(RandomWalkTask, mcts_config_dict, agent_name='MCTS1-test')
    mcts2 = build_MCTS_Agent(RandomWalkTask, mcts_config_dict, agent_name='MCTS2-test')

    expected_end_state = [3, 3]  # TODO, don't hardcode value

    trajectory = RandomWalkTask.run_episode([mcts1, mcts2], training=False)

    actual_end_state_p1 = trajectory[-1][-2][0]
    actual_end_state_p2 = trajectory[-1][-2][1]

    np.testing.assert_array_equal(expected_end_state, actual_end_state_p1)
    np.testing.assert_array_equal(expected_end_state, actual_end_state_p2)
