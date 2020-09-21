from test_fixtures import expert_iteration_config_dict, mcts_config_dict, Connect4Task

from torch.utils.tensorboard import SummaryWriter
import regym

from regym.util import extract_winner

from regym.tests.test_utils.play_against_fixed_opponent import learn_against_fix_opponent
from regym.tests.test_utils.parallel_play_against_fixed_opponent import parallel_learn_against_fix_opponent
from regym.rl_algorithms import rockAgent, build_Random_Agent, build_MCTS_Agent
from regym.rl_algorithms import build_ExpertIteration_Agent


def test_expert_iteration_can_take_actions_discrete_obvservation_discrete_action(Connect4Task, expert_iteration_config_dict):
    exIt_agent_1 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt1-test')
    exIt_agent_2 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt2-test')
    Connect4Task.run_episode([exIt_agent_1, exIt_agent_2], training=False)


def test_can_defeat_random_play_in_connect4_both_positions_single_env(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['mcts_budget'] = 100
    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='MCTS1-test')

    random_agent = build_Random_Agent(Connect4Task, {}, agent_name='Random')

    trajectory = Connect4Task.run_episode([ex_it, random_agent], training=False)
    assert extract_winner(trajectory) == 0  # First player (index 0) has a much higher budget

    trajectory = Connect4Task.run_episode([random_agent, ex_it], training=False)
    assert extract_winner(trajectory) == 1  # Second player (index 1) has a much higher budget


def test_can_defeat_random_play_in_connect4_both_positions_multi_env(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['mcts_budget'] = 100
    expert_iteration_config_dict['mcts_rollout_budget'] = 20
    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='MCTS1-test')

    random_agent = build_Random_Agent(Connect4Task, {}, agent_name='Random')

    trajectories = Connect4Task.run_episodes(
            [ex_it, random_agent], training=False, num_envs=4, num_episodes=4)

    assert all(map(lambda t: extract_winner(t) == 0, trajectories))  # First player (index 0) has a much higher budget

    trajectories = Connect4Task.run_episodes(
            [random_agent, ex_it], training=False, num_envs=4, num_episodes=4)
    assert all(map(lambda t: extract_winner(t) == 1, trajectories))  # Second player (index 1) has a much higher budget


def test_can_use_apprentice_in_expert_in_expansion_and_rollout_phase(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['use_apprentice_in_expert'] = True
    expert_iteration_config_dict['rollout_budget'] = 0
    exIt_agent_1 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt1-test')
    exIt_agent_2 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt2-test')
    Connect4Task.run_episode([exIt_agent_1, exIt_agent_2], training=False)


def test_train_apprentice_using_dagger_against_random_connect4(Connect4Task, expert_iteration_config_dict, mcts_config_dict):
    summary_writer = SummaryWriter('expert_iteration_test')
    regym.rl_algorithms.expert_iteration.expert_iteration_loss.summary_writer = summary_writer

    # Train worthy params
    expert_iteration_config_dict['use_apprentice_in_expert'] = False
    expert_iteration_config_dict['games_per_iteration'] = 10

    expert_iteration_config_dict['mcts_budget'] = 500
    expert_iteration_config_dict['mcts_rollout_budget'] = 100
    expert_iteration_config_dict['initial_memory_size'] = 10000
    expert_iteration_config_dict['memory_size_increase_frequency'] = 5
    expert_iteration_config_dict['end_memory_size'] = 30000
    expert_iteration_config_dict['use_dirichlet'] = False

    expert_iteration_config_dict['learning_rate'] = 1.0e-2
    expert_iteration_config_dict['batch_size'] = 256
    expert_iteration_config_dict['num_epochs_per_iteration'] = 4
    expert_iteration_config_dict['residual_connections'] = [(1, 2), (2, 3), (3, 4)]

    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt-test')

    random_agent = build_Random_Agent(Connect4Task, mcts_config_dict, agent_name=f"Random")

    parallel_learn_against_fix_opponent(ex_it,
            fixed_opponent=random_agent,
            agent_position=0,
            task=Connect4Task,
            training_episodes=5000,
            test_episodes=100,
            benchmarking_episodes=20,
            benchmark_every_n_episodes=500,
            reward_tolerance=0.2,
            maximum_average_reward=1.0,
            evaluation_method='last',
            show_progress=True,
            summary_writer=summary_writer)


def test_train_vanilla_exit_against_random_connect4(Connect4Task, expert_iteration_config_dict, mcts_config_dict):
    summary_writer = SummaryWriter('expert_iteration_test')
    regym.rl_algorithms.expert_iteration.expert_iteration_loss.summary_writer = summary_writer
    import torch
    torch.multiprocessing.set_start_method('forkserver')

    # Train worthy params
    expert_iteration_config_dict['use_apprentice_in_expert'] = True
    expert_iteration_config_dict['games_per_iteration'] = 100

    expert_iteration_config_dict['mcts_budget'] = 200
    expert_iteration_config_dict['mcts_rollout_budget'] = 10000
    expert_iteration_config_dict['initial_memory_size'] = 6000
    expert_iteration_config_dict['memory_size_increase_frequency'] = 2
    expert_iteration_config_dict['end_memory_size'] = 30000
    expert_iteration_config_dict['dirichlet_alpha'] = 1

    expert_iteration_config_dict['batch_size'] = 256
    expert_iteration_config_dict['num_epochs_per_iteration'] = 4
    # expert_iteration_config_dict['residual_connections'] = [(2, 3), (3, 4)]

    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name=f"ExIt-test:{expert_iteration_config_dict['mcts_budget']}")

    mcts_config_dict['budget'] = 1
    mcts_agent = build_MCTS_Agent(Connect4Task, mcts_config_dict, agent_name=f"MCTS:{mcts_config_dict['budget']}")

    parallel_learn_against_fix_opponent(ex_it,
            fixed_opponent=mcts_agent,
            agent_position=0,
            task=Connect4Task,
            training_episodes=5000,
            test_episodes=100,
            benchmarking_episodes=20,
            benchmark_every_n_episodes=500,
            reward_tolerance=0.2,
            maximum_average_reward=1.0,
            evaluation_method='last',
            show_progress=True,
            num_envs=torch.multiprocessing.cpu_count(),
            summary_writer=summary_writer)
