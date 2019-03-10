import os
import sys
sys.path.append(os.path.abspath('../../'))

import pytest
from environments.gym_parser import parse_gym_environment


@pytest.fixture
def ppo_config_dict():
    config = dict()
    config['discount'] = 0.995
    config['use_gae'] = True
    config['use_cuda'] = True
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 20
    config['mini_batch_size'] = 256
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-3
    config['adam_eps'] = 1.0e-5
    config['nbr_actor'] = 1
    config['horizon'] = 8192
    return config


@pytest.fixture
def ppo_config_dict_ma():
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = True
    config['use_cuda'] = False
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 15
    config['mini_batch_size'] = 256#4096
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['nbr_actor'] = 2#32
    config['horizon'] = 512
    return config


#@pytest.fixture
def ddpg_config_dict_ma():
    config = dict()
    config['discount'] = 0.99
    config['tau'] = 1e-3
    config['use_cuda'] = True
    config['nbrTrainIteration'] = 1 
    config['action_scaler'] = 1.0 
    config['use_HER'] = False
    config['HER_k'] = 2
    config['HER_strategy'] = 'future'
    config['HER_use_singlegoal'] = False 
    config['use_PER'] = True 
    config['PER_alpha'] = 0.7 
    config['replay_capacity'] = 25e3
    config['min_capacity'] = 5e3 
    config['batch_size'] = 32#128
    config['learning_rate'] = 3.0e-4
    config['nbr_actor'] = 4#32
    return config

@pytest.fixture
def dqn_config_dict():
    config = dict()
    config['learning_rate'] = 1.0e-3
    config['epsstart'] = 1
    config['epsend'] = 0.1
    config['epsdecay'] = 5.0e4
    config['double'] = False
    config['dueling'] = False
    config['use_cuda'] = True
    config['use_PER'] = False
    config['PER_alpha'] = 0.07
    config['min_memory'] = 1.e03
    config['memoryCapacity'] = 1.e03
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


#@pytest.fixture
def RoboSumoenv():
    import roboschool
    import gym
    return gym.make('RoboschoolSumo-v0')


#@pytest.fixture
def RoboSumoTask(RoboSumoenv):
    return parse_gym_environment(RoboSumoenv)
