from functools import partial, reduce

import pytest
import numpy as np

import torch
from test_fixtures import mcts_config_dict, Connect4Task, RandomWalkTask

from regym.rl_algorithms.agents import build_MCTS_Agent, build_Random_Agent, build_NeuralNet_Agent
from regym.networks.servers import request_prediction_from_server
from regym.rl_algorithms.MCTS import selection_strategies
from regym.rl_algorithms.MCTS import sequential_mcts
from regym.rl_algorithms.MCTS.sequential_node import SequentialNode
from regym.rl_algorithms.MCTS import util


def test_can_apply_dirichlet_noise_to_priors():
    seed = 420
    initial_priors = {0: 1/3, 1: 1/3, 2: 1/3}

    expected_changed_prior = {0: 1/3, 1: 1/3, 2: 1/3}

    np.random.seed(seed)
    actual_priors_1 = util.add_dirichlet_noise(alpha=1., p=initial_priors, noise_strength=1.)

    assert sum(actual_priors_1.values()) == 1.

    np.random.seed(seed)
    actual_priors_2 = util.add_dirichlet_noise(alpha=1., p=initial_priors, noise_strength=0.5)

    assert sum(actual_priors_2.values()) == 1.
    # The difference between initial_priors and actual_priors_1 should be greater than
    # the difference between initial_priors and actual_priors_2, due to reduced strength
    assert np.sum(np.abs(np.array(list(initial_priors.values())) - np.array(list(actual_priors_1.values())))) >\
           np.sum(np.abs(np.array(list(initial_priors.values())) - np.array(list(actual_priors_2.values()))))

    np.random.seed(seed)
    actual_priors_3 = util.add_dirichlet_noise(alpha=1., p=initial_priors, noise_strength=0)

    assert sum(actual_priors_3.values()) == 1.

    # If noise_strength = 0, then priors are unaltered
    np.testing.assert_array_equal(np.array(list(initial_priors.values())),  # All this casting is ugly
                                  np.array(list(actual_priors_3.values())))


def test_sequential_mcts_consistency_on_tree_properties_in_connect4(Connect4Task):
    '''
    - Root node should have the same number of visitations as the given budget.
    - A node's visit should be the sum of its children's visits
    TODO: add more nuance on what is being tested
    '''

    budget = 100
    rollout_budget = 100

    done = False
    observations = Connect4Task.env.reset()
    while not done:
        #p1_observation = observations[current_player]

        best_action, _, tree = sequential_mcts.MCTS(
            rootstate=Connect4Task.env,
            observation=None,
            budget=budget,
            rollout_budget=rollout_budget,
            selection_strat=selection_strategies.PUCT,
            exploration_factor=1.0,
            # It doesn't matter that we always act as player 0
            player_index=0,
            policy_fn=partial(util.random_selection_policy,
                              action_dim=Connect4Task.action_dim),
            evaluation_fn=None,
            use_dirichlet=False,
            dirichlet_alpha=None,
            dirichlet_strength=None,
            num_agents=2
        )
        _assert_only_top_node_is_root(tree)
        _assert_terminal_nodes_have_correct_values_in_deterministic_connect4(tree)
        # TODO: make function below work
        _assert_visitations_add_up(tree, budget, ['R'])

        _, _, done, _ = Connect4Task.env.step(best_action)


def _assert_only_top_node_is_root(node):
    ''' TODO: check and document '''
    if node.parent is None:
        assert node.is_root, ('Node with no parent (root node) should have '
                              'root property set')
    else:
        assert not(node.parent is None), 'Non-root node should have a parent node'
        assert not(node.is_root), 'Node should not be root'
    for child_node in node.children.values():
        _assert_only_top_node_is_root(child_node)


def _assert_visitations_add_up(node, budget, action_sequence):
    if node.is_root:
        assert node.N == budget, (
             'Root node should have the same number of '
             f'visitations as budget. Expected: {budget}. Actual: {node.N}')

    child_visitations = sum(node.N_a.values())
    if node.is_leaf and not node.is_terminal:
        assert node.N == 1, f'Leaf non-terminal nodes must only have been visited once. Visits {node.N}'
    if node.is_root:
        assert node.N == child_visitations, (
            'Root node child visitations should be the same as the sum '
            'of its children vistations. Which should be equal to the budget.\n'
            f'Sum N_a: {child_visitations}. N: {node.N}. Budget: {budget}.\n')
    elif not node.is_terminal:
        _assert_node_visitations_match_up_to_parent_edge_visitations(node)
        assert node.N == (1 + child_visitations), (
            'The visits of non-leaf, non-terminal nodes should be equal to the'
            ' sum of edge visits stored in that node + 1 (the one coming from'
            ' when the node was first visited as a leaf node).\n'
           f'Sum N_a: {child_visitations}. N: {node.N}\n'
           f'Branch path action sequence: {action_sequence}')

    for child_node in node.children.values():
        _assert_visitations_add_up(child_node, budget, action_sequence + [child_node.a])


def _assert_node_visitations_match_up_to_parent_edge_visitations(node):
    assert node.N == node.parent.N_a[node.a], (
        "The visitations stored on a node should match the visitations "
        "stored on the parent's edge leading up to this node\n"
        f"N: {node.N}. Parent's N_a: {node.parent.N_a[node.a]}")


def _assert_terminal_nodes_have_correct_values_in_deterministic_connect4(node):
     # Terminal nodes have value == N or -N (it is a deterministic game). Check that there are no draws?
     if node.is_terminal:
         total_node_value = node.parent.W_a[node.a]
         avg_node_value   = node.parent.Q_a[node.a]
         assert total_node_value == avg_node_value * node.N, (
             "In a deterministic environment, a terminal node's total and average "
             "value should be proportional to the number of visits.\n"
             f"Total {total_node_value}. Avg: {avg_node_value}. Visits {node.N}"
             )
         assert ((total_node_value == 0.) or \
                 (total_node_value == -node.N) or \
                 (total_node_value == node.N)), (
                     "In a deterministic environment with -1 (loss) / 0 (draw) / +1 (win) "
                     "final scores, a terminal node's value should be either zero or "
                     "equal to its visit count.\n"
                     f"Actual value: {total_node_value}. Visits {node.N}"
                 )

     for child_node in node.children.values():
         _assert_terminal_nodes_have_correct_values_in_deterministic_connect4(child_node)


def test_PUCT_selection_strategies_correctly_uses_priors():
    priors = {0: 1/4, 1: 1/2, 2: 1/4}

    node = SequentialNode(parent=None, player=0, a='R',
                          actions=[0,1,2],
                          priors=priors)  # This is what we care about

    expected_selecion_scores_values = [1/4,1/2,1/4]
    actual_selection_scores = selection_strategies.PUCT(node, c=1.)

    np.testing.assert_array_equal(
        expected_selecion_scores_values,
        list(actual_selection_scores.values()))


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
         'state_preprocess_fn': lambda x: torch.Tensor(x)
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
