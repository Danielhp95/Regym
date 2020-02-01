from test_fixtures import mcts_config_dict, Connect4Task
from utils import can_act_in_environment

from regym.rl_algorithms.agents import build_MCTS_Agent
from regym.util.play_matches import extract_winner

import pytest


def test_non_integer_budget_raises_value_error(Connect4Task):
    config = {'budget': 0.5}
    with pytest.raises(ValueError) as _:
        _ = build_MCTS_Agent(Connect4Task, config, 'name')

def test_mcts_can_take_actions_discrete_obvservation_discrete_action(Connect4Task, mcts_config_dict):
    mcts1 = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name='MCTS1-test')
    mcts2 = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name='MCTS1-test')
    Connect4Task.run_episode([mcts1, mcts2], training=False)


def test_can_defeat_random_play_in_connect4_both_positions(Connect4Task, mcts_config_dict):
    mcts1 = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name='MCTS1-test')
    mcts_config_dict['budget'] = 50
    mcts2 = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name='MCTS1-test')
    trajectory = Connect4Task.run_episode([mcts1, mcts2], training=False)

    assert extract_winner(trajectory) == 1 # Second player (index 1) has a much higer budget
    trajectory = Connect4Task.run_episode([mcts2, mcts1], training=False)
    assert extract_winner(trajectory) == 0 # First player (index 0) has a much higer budget
