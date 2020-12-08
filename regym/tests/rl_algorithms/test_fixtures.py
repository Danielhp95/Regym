from math import sqrt
import pytest
from regym.environments import generate_task
from regym.environments import EnvType


@pytest.fixture
def sac_config_dict():
    config = dict()
    config['learning_rate'] = 3e-3
    config['memory_size'] = 20000
    config['gamma'] = 0.99
    config['tau'] = 0.995
    config['batch_size'] = 32
    config['alpha'] = 0.01
    config['update_after'] = 500
    config['update_every'] = 1
    config['use_cuda'] = False
    return config


@pytest.fixture
def ppo_config_dict():
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = False
    config['use_cuda'] = False
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 32
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['horizon'] = 128
    config['phi_arch'] = 'MLP'
    config['actor_arch'] = 'None'
    config['critic_arch'] = 'None'
    return config


@pytest.fixture
def ppo_rnn_config_dict():
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = False
    config['use_cuda'] = False
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 32
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['horizon'] = 256
    config['phi_arch'] = 'RNN'
    config['actor_arch'] = 'None'
    config['critic_arch'] = 'None'
    return config


@pytest.fixture
def dqn_config_dict():
    config = dict()
    config['learning_rate'] = 1.0e-5
    config['epsstart'] = 0.4
    config['epsend'] = 0.01
    config['epsdecay'] = 5.0e3
    config['double'] = False
    config['dueling'] = False
    config['use_cuda'] = False
    config['use_PER'] = False
    config['PER_alpha'] = 0.07
    config['min_memory'] = 2.e03
    config['memoryCapacity'] = 2.e03
    config['nbrTrainIteration'] = 8
    config['batch_size'] = 256
    config['gamma'] = 0.99
    config['tau'] = 1.0e-2
    return config


@pytest.fixture
def tabular_q_learning_config_dict():
    config = dict()
    config['learning_rate'] = 0.9
    config['discount_factor'] = 0.99
    config['epsilon_greedy'] = 0.1
    config['use_repeated_update_q_learning'] = False
    config['temperature'] = 1
    return config


@pytest.fixture
def reinforce_config_dict():
    config = dict()
    config['learning_rate'] = 5.0e-3
    config['episodes_before_update'] = 50 # Do not make less than 2, for reinforce_test.py
    config['adam_eps'] = 1.0e-5
    return config


@pytest.fixture
def a2c_config_dict():
    config = dict()
    config['discount_factor'] = 0.9
    config['n_steps'] = 5
    config['samples_before_update'] = 30
    config['learning_rate'] = 1.0e-3
    config['adam_eps'] = 1.0e-5
    return config


@pytest.fixture
def mcts_config_dict():
    config = dict()
    config['budget'] = 2
    config['rollout_budget'] = 100000
    config['selection_phase'] = 'ucb1'
    config['exploration_factor_puct'] = 4
    config['exploration_factor_ucb1'] = sqrt(2)
    config['use_dirichlet'] = True
    config['dirichlet_alpha'] = sqrt(2)
    return config


@pytest.fixture
def expert_iteration_config_dict():
    config = dict()
    # Higher level ExIt params
    config['use_agent_modelling'] = False
    config['use_agent_modelling_in_mcts'] = False
    config['use_apprentice_in_expert'] = False
    config['games_per_iteration'] = 2
    # Dataset params
    config['initial_memory_size'] = 3
    config['memory_size_increase_frequency'] = 1
    config['end_memory_size'] = 9
    # MCTS config
    config['mcts_budget'] = 20
    config['mcts_rollout_budget'] = 0
    config['mcts_exploration_factor'] = sqrt(2)
    config['mcts_use_dirichlet'] = True
    config['mcts_dirichlet_alpha'] = sqrt(2)
    # Neural net config
    config['batch_size'] = 2
    config['num_epochs_per_iteration'] = 2
    config['learning_rate'] = 1.0e-3
    # NN: Feature extractor
    config['feature_extractor_arch'] = 'CNN'
    config['use_batch_normalization'] = False
    config['preprocessed_input_dimensions'] = [7, 6]  # To play Connect 4
    config['channels'] = [3, 10, 11, 12, 13, 1]  # To play Connect 4
    config['kernel_sizes'] = [3, 3, 3, 3, 3]  # To play Connect 4
    config['paddings'] = [1, 1, 1, 1, 1]  # To play Connect 4
    config['strides'] = [1, 1, 1, 1, 1]  # To play Connect 4
    config['residual_connections'] = []
    config['critic_gate_fn'] = 'tanh'
    return config


@pytest.fixture
def FrozenLakeTask(): # Discrete Action / Observation space
    return generate_task('FrozenLake-v0')


@pytest.fixture
def CartPoleTask(): # Discrete Action / Continuous Observation space
    return generate_task('CartPole-v0')


@pytest.fixture
def PendulumTask(): # Continuous Action / Observation space
    return generate_task('Pendulum-v0')


@pytest.fixture
def RPSTask():
    import gym_rock_paper_scissors
    return generate_task('RockPaperScissors-v0', EnvType.MULTIAGENT_SIMULTANEOUS_ACTION)


@pytest.fixture
def RPSTaskSingleRepetition():
    import gym_rock_paper_scissors
    return generate_task('RockPaperScissors-v0', EnvType.MULTIAGENT_SIMULTANEOUS_ACTION,
                         max_repetitions=1, stacked_observations=3)


@pytest.fixture
def KuhnTask():
    import gym_kuhn_poker
    return generate_task('KuhnPoker-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)


@pytest.fixture
def Connect4Task():
    import gym_connect4
    return generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)


@pytest.fixture
def RandomWalkTask():
    from gym.envs.registration import register
    register(id='RandomWalk-v0', entry_point='regym.tests.rl_algorithms.random_walk_env:RandomWalkEnv')
    return generate_task('RandomWalk-v0', EnvType.MULTIAGENT_SIMULTANEOUS_ACTION)
