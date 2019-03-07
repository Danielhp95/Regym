import os
import sys
sys.path.append(os.path.abspath('..'))
from training_schemes import NaiveSelfPlay, HalfHistorySelfPlay, FullHistorySelfPlay

from rl_algorithms import build_DQN_Agent
from rl_algorithms import build_TabularQ_Agent
from rl_algorithms import build_PPO_Agent
from rl_algorithms import rockAgent, paperAgent, scissorsAgent

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
    def partial_match_build_function(agent_name, task, config):
        if agent_name.startswith('tabularqlearning'): return build_TabularQ_Agent(task, config, agent_name)
        if agent_name.startswith('deepqlearning'): return build_DQN_Agent(task, config, agent_name)
        if agent_name.startswith('ppo'): return build_PPO_Agent(task, config, agent_name)
        else: raise ValueError('Unkown agent name: {agent_name}'.format(agent_name))
    task = environments.parse_gym_environment(environment)
    return [partial_match_build_function(agent, task, config) for agent, config in agent_configurations.items()]


def initialize_fixed_agents(fixed_agents):
    '''
    Builds a fixed (stationary) agent for each agent in :param: fixed_agents.
    ASSUMPTION: Each agent is able to take actions in the environment that will be used for the experiment
    :param: List of requested fixed agent names to be created
    :return: array of initialized stationary agents
    '''
    fix_agent_build_functions = {'rockagent': rockAgent, 'paperagent': paperAgent, 'scissorsagent': scissorsAgent}
    check_for_unknown_candidate_input(fix_agent_build_functions.keys(), fixed_agents, 'fixed_agents')
    return [fix_agent_build_functions[agent.lower()] for agent in fixed_agents]


def filter_relevant_agent_configurations(experiment_config, agents_config):
    '''
    The config file allows to have configuration for RL algorithms that will not be used.
    This allows to keep all configuration in a single file.
    The configuration that will be used is explicitly captured in :param: experiment_config
    '''
    return {agent: config for agent, config in agents_config.items()
            if any(map(lambda algorithm: agent.startswith(algorithm), experiment_config['algorithms']))}
