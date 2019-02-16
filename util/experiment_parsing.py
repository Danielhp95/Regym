import os
import sys
sys.path.append(os.path.abspath('..'))
from training_schemes import NaiveSelfPlay, HalfHistorySelfPlay, FullHistorySelfPlay

from rl_algorithms import build_DQN_Agent
from rl_algorithms import build_TabularQ_Agent
# from rl_algorithms import build_PPO_Agent
from rl_algorithms import rockAgent, paperAgent, scissorsAgent
from rl_algorithms import AgentHook

import environments


def initialize_training_schemes(training_schemes_cli):
    self_play_training_schemes = {'fullhistoryselfplay': FullHistorySelfPlay, 'halfhistoryselfplay': HalfHistorySelfPlay, 'naiveselfplay': NaiveSelfPlay}
    return [self_play_training_schemes[t_s.lower()] for t_s in training_schemes_cli]


def initialize_algorithms(environment, agent_configurations):
    '''
    TODO document
    '''

    task = environments.parse_gym_environment(environment)
    agent_build_functions = {'tabularqlearning': build_TabularQ_Agent, 'deepqlearning': build_DQN_Agent}
    return [agent_build_functions[agent](task, config) for agent, config in agent_configurations.items()]


def find_paths(algorithms, base_path):
    return [os.path.join(base_path, algorithm.lower())+'.pt' for algorithm in algorithms]


def initialize_fixed_agents(fixed_agents_cli):
    '''
    TODO test
    '''
    fix_agent_build_functions = {'rockagent': rockAgent, 'paperagent': paperAgent, 'scissorsagent': scissorsAgent}
    return [AgentHook(fix_agent_build_functions[agent.lower()]) for agent in fixed_agents_cli]
