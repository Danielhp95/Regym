import os
import sys
sys.path.append(os.path.abspath('../../'))

from rl_algorithms.agents import MixedStrategyAgent
from rl_algorithms.agents import build_TabularQ_Agent, TabularQLearningAgent
from rl_algorithms.agents import build_DQN_Agent, DeepQNetworkAgent 
from rl_algorithms.agents import build_DDPG_Agent, DDPGAgent 
from rl_algorithms.agent_hook import AgentHook
import numpy as np
import pytest


@pytest.fixture
def env():
    class Env():
        state_space_size  = 5
        action_space_size = 5
        hashing_function = lambda x: x
    return Env()


def test_agents_instantiation(env):
    tql = build_TabularQ_Agent()
    dqn = build_DQN_Agent()
    ddpg = build_DDPG_Agent()

def test_agent_hooking(env=None):
    tql = build_TabularQ_Agent()
    dqn = build_DQN_Agent()
    ddpg = build_DDPG_Agent()
    
    path_tql = './data/tql'
    ah_tql = AgentHook(tql,path=path_tql)
    path_dqn = './data/dqn'
    ah_dqn = AgentHook(dqn,path=path_dqn)
    path_ddpg = './data/ddpg'
    ah_ddpg = AgentHook(ddpg,path=path_ddpg)

    tql2 = ah_tql()
    dqn2 = ah_dqn()
    ddpg2 = ah_ddpg()

    assert tql == tql2 
    tests = [dqn.kwargs[el] == dqn2.kwargs[el] for el in dqn.kwargs if not('model' in el) ]
    for el in tests : assert(el)
    tests = [ddpg.kwargs[el] == ddpg2.kwargs[el] for el in ddpg.kwargs if not('model' in el or 'process' in el) ]
    for el in tests : assert(el)
    