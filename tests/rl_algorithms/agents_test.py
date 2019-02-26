import os
import sys
sys.path.append(os.path.abspath('../../'))

from rl_algorithms.agents import MixedStrategyAgent
from rl_algorithms.agents import build_TabularQ_Agent, TabularQLearningAgent
from rl_algorithms.agents import build_DQN_Agent, DeepQNetworkAgent 
from rl_algorithms.agents import build_DDPG_Agent, DDPGAgent 
from rl_algorithms.agent_hook import AgentHook
from environments.gym_parser import parse_gym_environment
import numpy as np
import pytest

@pytest.fixture
def RPSenv():
    import gym
    import gym_rock_paper_scissors
    return gym.make('RockPaperScissors-v0')


@pytest.fixture
def RPSTask(RPSenv):
    return parse_gym_environment(RPSenv)


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
    return config

@pytest.fixture
def dqn_config_dict():
    config = dict()
    config['batch_size'] = 32
    config['gamma'] = 0.99
    config['tau'] = 1.0e-3
    config['learning_rate'] = 1.0e-3
    config['epsstart'] = 0.8
    config['epsend'] = 0.05
    config['epsdecay'] = 1.0e3
    config['double'] = False
    config['dueling'] = False
    config['use_cuda'] = False
    config['use_PER'] = False
    config['PER_alpha'] = 0.07
    config['min_memory'] = 5.0e1
    config['memoryCapacity'] = 25.0e3
    config['nbrTrainIteration'] = 32
    return config

def test_agent_hooking(RPSenv, RPSTask, dqn_config_dict,ppo_config_dict):
    #tql = build_TabularQ_Agent()
    dqn = build_DQN_Agent(RPSTask,dqn_config_dict)
    #ddpg = build_DDPG_Agent(RPSTask,ppo_config_dict)

    #path_tql = './data/tql'
    #ah_tql = AgentHook(tql,path=path_tql)
    path_dqn = './data/dqn'
    ah_dqn = AgentHook(dqn,path=path_dqn)
    #path_ddpg = './data/ddpg'
    #ah_ddpg = AgentHook(ddpg,path=path_ddpg)

    #tql2 = ah_tql()
    dqn2 = ah_dqn()
    #ddpg2 = ah_ddpg()

    #assert tql == tql2 
    
    tests = [dqn.kwargs[el] == dqn2.kwargs[el] for el in dqn.kwargs if not('model' in el) ]
    import ipdb; ipdb.set_trace()
    assert(all(tests))
    
    #tests = [ddpg.kwargs[el] == ddpg2.kwargs[el] for el in ddpg.kwargs if not('model' in el or 'process' in el) ]
    #for el in tests : assert(el)
    