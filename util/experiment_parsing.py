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
    '''
    Error checking. Checks that all :param: candidates have valid :known: functions
    :param known: valid / implemented string names
    :param candidates: candidate string names
    :param category_name: String identifying the category of candidates
    :raises ValueError: if unknown candidates are found
    '''
    unknown_candidates = list(filter(lambda x: x not in known, candidates))
    if len(unknown_candidates) > 0:
        raise ValueError('Unknown {}(s): {}. Valid candidates are: {}'.format(category_name, unknown_candidates, known))


def initialize_training_schemes(candidate_training_schemes):
    '''
    Creates a list containing pointers to the relevant self_play training scheme functions
    :param candidate_training_schemes: requested training schemes
    :return: list containing pointers to the corresponding self_play training schemes functions
    '''
    self_play_training_schemes = {'fullhistoryselfplay': FullHistorySelfPlay, 'halfhistoryselfplay': HalfHistorySelfPlay, 'naiveselfplay': NaiveSelfPlay}
    check_for_unknown_candidate_input(self_play_training_schemes.keys(), candidate_training_schemes, 'training schemes')
    return [self_play_training_schemes[t_s.lower()] for t_s in candidate_training_schemes]


def initialize_algorithms(environment, agent_configurations):
    '''
    Builds an agent for each agent in :param: agent_configurations
    suitable to act and process experience from :param: environment
    :param environment: environment on which the agents will act
    :param agent_configurations: configuration dictionaries for each requested agent
    :returns: array of agents built according to their corresponding configuration dictionaries
    '''
    task = environments.parse_gym_environment(environment)
    agent_build_functions = {'tabularqlearning': build_TabularQ_Agent, 'deepqlearning': build_DQN_Agent}
    check_for_unknown_candidate_input(agent_build_functions.keys(), agent_configurations.keys(), 'agent')
    return [agent_build_functions[agent](task, config) for agent, config in agent_configurations.items()]


def find_paths(algorithms, base_path):
    '''
    Creates path based on algorithm names
    :param algorithms: List of algorithm names
    :param base_path: string path TODO: figure what it is
    :returns: list of paths, one for each algorithm
    '''
    return [os.path.join(base_path, algorithm.lower())+'.pt' for algorithm in algorithms]


def initialize_fixed_agents(fixed_agents):
    '''
    Builds a fixed (stationary) agent for each agent in :param: fixed_agents.
    ASSUMPTION: Each agent is able to take actions in the environment that will be used for the experiment
    :param: List of requested fixed agent names to be created
    :return: array of initialized stationary agents
    '''
    fix_agent_build_functions = {'rockagent': rockAgent, 'paperagent': paperAgent, 'scissorsagent': scissorsAgent}
    check_for_unknown_candidate_input(fix_agent_build_functions.keys(), fixed_agents, 'fixed_agents')
    return [AgentHook(fix_agent_build_functions[agent.lower()]) for agent in fixed_agents]
