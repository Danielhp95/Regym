from test_fixtures import expert_iteration_config_dict, Connect4Task

from regym.util import extract_winner

from play_against_fixed_opponent import learn_against_fix_opponent
from regym.rl_algorithms import rockAgent, build_Random_Agent
from regym.rl_algorithms import build_ExpertIteration_Agent


def test_expert_iteration_can_take_actions_discrete_obvservation_discrete_action(Connect4Task, expert_iteration_config_dict):
    exIt_agent_1 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt1-test')
    exIt_agent_2 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt2-test')
    Connect4Task.run_episode([exIt_agent_1, exIt_agent_2], training=True)


def test_can_defeat_random_play_in_connect4_both_positions(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['mcts_budget'] = 100
    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='MCTS1-test')

    random_agent = build_Random_Agent(Connect4Task, {}, agent_name='Random')

    trajectory = Connect4Task.run_episode([ex_it, random_agent], training=False)
    assert extract_winner(trajectory) == 0  # First player (index 0) has a much higher budget

    trajectory = Connect4Task.run_episode([random_agent, ex_it], training=False)
    assert extract_winner(trajectory) == 1  # Second player (index 1) has a much higher budget


def test_train_against_random_connect4(Connect4Task, expert_iteration_config_dict):
    # TODO: ADD max size
    # TODO: ADD num_episodes per update
    # TODO: add functionality to remove older episodes once max episodes is reached
    from torch.utils.tensorboard import SummaryWriter
    import regym
    regym.rl_algorithms.expert_iteration.expert_iteration_loss.summary_writer = SummaryWriter('expert_iteration_test')

    expert_iteration_config_dict['mcts_budget'] = 100
    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExItvsRandom-test')

    random_agent = build_Random_Agent(Connect4Task, {}, agent_name='Random')

    learn_against_fix_opponent(ex_it, fixed_opponent=random_agent,
                               agent_position=0,
                               task=Connect4Task,
                               total_episodes=13000, training_percentage=0.9,
                               reward_tolerance=0.2,
                               maximum_average_reward=1.0,
                               evaluation_method='last')
