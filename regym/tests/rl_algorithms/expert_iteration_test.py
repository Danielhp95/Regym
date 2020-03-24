from test_fixtures import expert_iteration_config_dict, Connect4Task

from regym.rl_algorithms import rockAgent
from regym.rl_algorithms import build_ExpertIteration_Agent


def test_expert_iteration_can_take_actions_discrete_obvservation_discrete_action(Connect4Task, expert_iteration_config_dict):
    exIt_agent_1 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt1-test')
    exIt_agent_2 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt2-test')
    Connect4Task.run_episode([exIt_agent_1, exIt_agent_2], training=False)


#def test_can_defeat_random_play_in_connect4_both_positions(Connect4Task, expert_iteration_config_dict, mcts_config_dict):
#    mcts1 = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name='MCTS1-test')
#    mcts_config_dict['budget'] = 1
#    random_agent = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name='Random-MCTS')
#    trajectory = Connect4Task.run_episode([mcts1, random_agent], training=False)
#
#    assert extract_winner(trajectory) == 1  # Second player (index 1) has a much higher budget
#    trajectory = Connect4Task.run_episode([random_agent, mcts1], training=False)
#    assert extract_winner(trajectory) == 0  # First player (index 0) has a much higher budget
