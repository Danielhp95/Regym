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


def check_for_unknown_candidate_input(known, candidates, category_name):
    unknown_candidates = list(filter(lambda x: x not in known, candidates))
    if len(unknown_candidates) > 0:
        raise ValueError('Unknown {}(s): {}. Valid candidates are: {}'.format(category_name, unknown_candidates, known))


def initialize_training_schemes(candidate_training_schemes):
    self_play_training_schemes = {'fullhistoryselfplay': FullHistorySelfPlay, 'halfhistoryselfplay': HalfHistorySelfPlay, 'naiveselfplay': NaiveSelfPlay}
    check_for_unknown_candidate_input(self_play_training_schemes.keys(), candidate_training_schemes, 'training schemes')
    return [self_play_training_schemes[t_s.lower()] for t_s in candidate_training_schemes]


def initialize_algorithms(environment, agent_configurations):
    '''
    TODO document
    '''

    task = environments.parse_gym_environment(environment)
    agent_build_functions = {'tabularqlearning': build_TabularQ_Agent, 'deepqlearning': build_DQN_Agent}
    check_for_unknown_candidate_input(agent_build_functions.keys(), agent_configurations.keys(), 'agent')
    return [agent_build_functions[agent](task, config) for agent, config in agent_configurations.items()]


def find_paths(algorithms, base_path):
    return [os.path.join(base_path, algorithm.lower())+'.pt' for algorithm in algorithms]


def initialize_fixed_agents(fixed_agents):
    '''
    TODO test
    '''
    fix_agent_build_functions = {'rockagent': rockAgent, 'paperagent': paperAgent, 'scissorsagent': scissorsAgent}
    check_for_unknown_candidate_input(fix_agent_build_functions.keys(), fixed_agents, 'fixed_agents')
    return [AgentHook(fix_agent_build_functions[agent.lower()]) for agent in fixed_agents]
