from typing import Callable

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from test_fixtures import expert_iteration_config_dict, mcts_config_dict, Connect4Task
import regym
from regym.tests.test_utils.play_against_fixed_opponent import learn_against_fix_opponent
from regym.tests.test_utils.parallel_play_against_fixed_opponent import parallel_learn_against_fix_opponent
from regym.rl_algorithms import rockAgent, build_Random_Agent, build_MCTS_Agent, build_Deterministic_Agent
from regym.rl_algorithms import build_ExpertIteration_Agent
from regym.rl_algorithms.agents import ExpertIterationAgent
from regym.rl_algorithms.replay_buffers import Storage


def test_expert_iteration_can_take_actions_discrete_obvservation_discrete_action(Connect4Task, expert_iteration_config_dict):
    exIt_agent_1 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt1-test')
    exIt_agent_2 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt2-test')
    Connect4Task.run_episode([exIt_agent_1, exIt_agent_2], training=False)


def test_can_defeat_random_play_in_connect4_both_positions_single_env(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['mcts_budget'] = 100
    expert_iteration_config_dict['mcts_rollout_budget'] = 20
    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='MCTS1-test')

    random_agent = build_Random_Agent(Connect4Task, {}, agent_name='Random')

    trajectory = Connect4Task.run_episode([ex_it, random_agent], training=False)
    assert trajectory.winner == 0  # First player (index 0) has a much higher budget

    trajectory = Connect4Task.run_episode([random_agent, ex_it], training=False)
    assert trajectory.winner == 1  # Second player (index 1) has a much higher budget


def test_can_defeat_random_play_in_connect4_both_positions_multi_env(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['mcts_budget'] = 100
    expert_iteration_config_dict['mcts_rollout_budget'] = 20
    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='MCTS1-test')

    random_agent = build_Random_Agent(Connect4Task, {}, agent_name='Random')

    trajectories = Connect4Task.run_episodes(
            [ex_it, random_agent], training=False, num_envs=4, num_episodes=4)

    assert all(map(lambda t: t.winner == 0, trajectories))  # First player (index 0) has a much higher budget

    trajectories = Connect4Task.run_episodes(
            [random_agent, ex_it], training=False, num_envs=4, num_episodes=4)
    assert all(map(lambda t: t.winner == 1, trajectories))  # Second player (index 1) has a much higher budget


def test_can_collect_opponent_action_distributions_multi_env(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['use_agent_modelling'] = True
    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt-opponent_modelling-test')
    assert ex_it.requires_opponents_prediction

    random_agent = build_Random_Agent(Connect4Task, {}, agent_name='Random')

    _ = Connect4Task.run_episodes(
        agent_vector=[ex_it, random_agent],
        training=True,  # Required for ExIt agent to `handle_experience`s
        num_envs=2, num_episodes=2)
    # We only check for existance of the key, rather than it's content
    assert 'opponent_policy' in ex_it.algorithm.memory.keys
    assert 'opponent_s' in ex_it.algorithm.memory.keys
    # ex_it.algorithm.memory. Once you fix it. push!
    assert len(ex_it.algorithm.memory.opponent_policy) == len(ex_it.algorithm.memory.s)
    assert len(ex_it.algorithm.memory.opponent_policy) == len(ex_it.algorithm.memory.opponent_s)


def test_can_query_learnt_opponent_models_at_train_time(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['use_apprentice_in_expert'] = True
    expert_iteration_config_dict['use_learnt_opponent_models_in_mcts'] = True

    ex_it = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict,
                                        agent_name='ExIt-opponent_modelling-test')
    assert ex_it.use_apprentice_in_expert
    assert ex_it.use_learnt_opponent_models_in_mcts
    # TODO: figure out a programatic way of testing this.


def test_agent_initially_configured_to_use_true_opponent_models_can_switch_to_using_learnt_models_in_mcts(Connect4Task, expert_iteration_config_dict):
    torch.multiprocessing.set_start_method('spawn')
    expert_iteration_config_dict['use_agent_modelling'] = True
    expert_iteration_config_dict['use_apprentice_in_expert'] = True
    expert_iteration_config_dict['use_true_agent_models_in_mcts'] = True

    brexit_agent = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict,
                                               agent_name='ExIt-opponent_modelling-test')
    rando = build_Random_Agent(Connect4Task, {}, 'Rando')
    assert brexit_agent.use_apprentice_in_expert
    assert brexit_agent.use_true_agent_models_in_mcts
    assert not brexit_agent.use_learnt_opponent_models_in_mcts

    brexit_agent.use_learnt_opponent_models_in_mcts = True
    assert brexit_agent.use_apprentice_in_expert
    assert brexit_agent.use_learnt_opponent_models_in_mcts
    assert not brexit_agent.use_true_agent_models_in_mcts
    # TODO: figure out a programatic way of testing this.
    # Currently only checking if crashes. One can internally
    # check that MCTSAgent is using the policy function 
    # learnt_opponent_aware_server_based_policy_fn().
    # Which is bad, but deadlines are always coming, aren't they.
    Connect4Task.run_episodes([brexit_agent, rando],
                              num_episodes=1, num_envs=1,
                              training=False)


def test_apprentice_can_model_expert(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['use_agent_modelling'] = False

    expert_iteration_config_dict['batch_size'] = 10
    expert_iteration_config_dict['num_epochs_per_iteration'] = 10

    _test_learn_against_fixed_distribution(
        Connect4Task,
        expert_iteration_config_dict,
        prediction_process_fn=lambda p: p['probs']
    )


def test_can_model_fixed_stochastic_policy(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['use_agent_modelling'] = True

    expert_iteration_config_dict['batch_size'] = 10
    expert_iteration_config_dict['num_epochs_per_iteration'] = 10

    _test_learn_against_fixed_distribution(
        Connect4Task,
        expert_iteration_config_dict,
        prediction_process_fn=lambda p: p['policy_0']
    )


def _test_learn_against_fixed_distribution(task, config, prediction_process_fn: Callable):
    # Random stochstic policy
    target_policy = torch.Tensor([0.17, 0.03, 0.05, 0.4, 0.05, 0.2, 0.1])

    # Create agent and dataset
    memory = create_memory(100, target_policy, task)
    ex_it = build_ExpertIteration_Agent(task, config, agent_name='ExIt-opponent_modelling-test')
    ex_it.algorithm.memory = memory

    # Train
    num_trains = 4
    for _ in range(num_trains):
        ex_it.algorithm.train(ex_it.apprentice)

    # Test
    _test_model(ex_it, target_policy, task, prediction_process_fn)


def _test_model(agent, target_policy, task, prediction_process_fn,
                test_iterations=50, kl_divergence_tolerance=0.01):
    for i in range(test_iterations):
        with torch.no_grad():
            sample_obs = agent.state_preprocess_fn(
                task.env.observation_space.sample()[0]
            )
            processed_prediction = prediction_process_fn(agent.apprentice(sample_obs))
            kl_div = torch.nn.functional.kl_div(
                processed_prediction.log(),
                target_policy
            )
            assert kl_div.item() < kl_divergence_tolerance, (f'Prection was too far off from allowed tolerance {kl_divergence_tolerance}\n'
                                                             f'Target distribution: {target_policy}\n'
                                                             f'Prediction: {processed_prediction}\n'
                                                             f'Diff: {torch.abs(target_policy - processed_prediction)}')


def test_can_use_apprentice_in_expert_in_expansion_and_rollout_phase(Connect4Task, expert_iteration_config_dict):
    expert_iteration_config_dict['use_apprentice_in_expert'] = True
    expert_iteration_config_dict['rollout_budget'] = 0
    exIt_agent_1 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt1-test')
    exIt_agent_2 = build_ExpertIteration_Agent(Connect4Task, expert_iteration_config_dict, agent_name='ExIt2-test')
    Connect4Task.run_episode([exIt_agent_1, exIt_agent_2], training=False)


def test_train_apprentice_using_dagger_against_random_connect4(Connect4Task, expert_iteration_config_dict, mcts_config_dict):
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
    ex_it.algorithm.summary_writer = SummaryWriter('expert_iteration_test')

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


def create_memory(size: int,
                  fixed_policy_target: torch.Tensor,
                  task) -> Storage:
    memory = Storage(size, keys=['opponent_policy', 'opponent_s', 's', 'V', 'normalized_child_visitations'])
    for _ in range(size):
        random_opponent_s = torch.rand((1, *task.observation_dim))
        random_s = torch.rand((1, *task.observation_dim))
        memory.add({'s': random_s,
                    'V': torch.Tensor([0.]),
                    'normalized_child_visitations': fixed_policy_target,
                    'opponent_s': random_opponent_s,
                    'opponent_policy': fixed_policy_target})
    return memory


def test_train_vanilla_exit_against_random_connect4(Connect4Task, expert_iteration_config_dict, mcts_config_dict):
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
    ex_it.algorithm.summary_writer = SummaryWriter('expert_iteration_test')

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
