import pytest
import numpy as np

import torch
from test_fixtures import mcts_config_dict, Connect4Task, RandomWalkTask

from regym.rl_algorithms.agents import build_MCTS_Agent, build_Random_Agent, build_NeuralNet_Agent
from regym.networks.servers import request_prediction_from_server



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
    mcts_config_dict['rollout_budget'] = 100
    win_task_in_both_positions(Connect4Task, mcts_config_dict) # First player (index 0) has a much higher budget


def test_can_defeat_random_play_in_connect4_both_positions_using_puct(Connect4Task, mcts_config_dict):
    mcts_config_dict['selection_phase'] = 'puct'
    mcts_config_dict['budget'] = 50
    mcts_config_dict['rollout_budget'] = 100
    win_task_in_both_positions(Connect4Task, mcts_config_dict)  # First player (index 0) has a much higher budget


def win_task_in_both_positions(task, mcts_config_dict):
    mcts = build_MCTS_Agent(task, mcts_config_dict, agent_name='MCTS1-test')
    random = build_Random_Agent(task, {}, agent_name='Random-test')
    t1 = task.run_episode([mcts, random], training=False)

    assert t1.winner == 0  # First player (index 0) has a much higher budget
    t2 = task.run_episode([random, mcts], training=False)
    assert t2.winner == 1  # Second_player (index 1) has a much higher budget


def test_can_coordinate_in_simulatenous_random_walk(RandomWalkTask, mcts_config_dict):
    mcts_config_dict['budget'] = 50
    mcts_config_dict['rollout_budget'] = 0

    mcts1 = build_MCTS_Agent(RandomWalkTask, mcts_config_dict, agent_name='MCTS1-test')
    mcts2 = build_MCTS_Agent(RandomWalkTask, mcts_config_dict, agent_name='MCTS2-test')

    expected_end_state = [3, 3]  # TODO, don't hardcode value

    trajectory = RandomWalkTask.run_episode([mcts1, mcts2], training=False)

    actual_end_state_p1 = trajectory[-1].succ_observation[0]
    actual_end_state_p2 = trajectory[-1].succ_observation[1]

    np.testing.assert_array_equal(expected_end_state, actual_end_state_p1)
    np.testing.assert_array_equal(expected_end_state, actual_end_state_p2)


def test_can_query_true_opponent_model(Connect4Task, mcts_config_dict):
    #expert_iteration_config_dict['use_agent_modelling_in_mcts'] = True
    mcts_agent = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name='MCTS-test')

    expected_opponent_policy = np.array([0., 1., 0.])

    dummy_nn_agent = build_NeuralNet_Agent(
        Connect4Task,
        {'neural_net': _generate_dummy_neural_net(torch.from_numpy(
            expected_opponent_policy).unsqueeze(0)),
         'preprocessing_fn': lambda x: torch.Tensor(x)
        },
        'Test-NNAgent')

    # Allow MCTSAgent to create a server
    mcts_agent.access_other_agents([dummy_nn_agent], Connect4Task, num_envs=1)
    assert mcts_agent.opponent_server_handler, 'Should be present and contain a neural_net_server handler'

    # Request opponent predictions from server
    connection = mcts_agent.opponent_server_handler.client_connections[0]
    actual_opponent_policy = request_prediction_from_server(0., 0., connection, key='policy_0')

    np.testing.assert_array_equal(expected_opponent_policy, actual_opponent_policy)


def _generate_dummy_neural_net(fixed_return_value: torch.Tensor):
    class DummyNet(torch.nn.Module):
        def __init__(self, fixed_return_value: torch.Tensor):
            super().__init__()
            self.fixed_return_value = fixed_return_value

        def forward(self, x, legal_actions=None):
            # The key to this dictionary is Regym convention!
            return {'policy_0': fixed_return_value}
    return DummyNet(fixed_return_value)
