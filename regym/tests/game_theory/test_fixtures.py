import pytest
from regym.environments import generate_task, EnvType


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
    config['mini_batch_size'] = 256
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['horizon'] = 1024
    config['phi_arch'] = 'MLP'
    config['actor_arch'] = 'None'
    config['critic_arch'] = 'None'
    return config

@pytest.fixture
def RPSTask():
    import gym_rock_paper_scissors
    return generate_task('RockPaperScissors-v0', EnvType.MULTIAGENT_SIMULTANEOUS_ACTION)


@pytest.fixture
def pendulum_task():
    return generate_task('Pendulum-v0')
