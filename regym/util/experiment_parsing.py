from functools import partial
from typing import Dict

import numpy as np

from regym.training_schemes import NaiveSelfPlay
from regym.training_schemes import PSRONashResponse
from regym.training_schemes import DeltaDistributionalSelfPlay, DeltaLimitDistributionalSelfPlay

from regym.rl_algorithms import build_DQN_Agent
from regym.rl_algorithms import build_TabularQ_Agent
from regym.rl_algorithms import build_PPO_Agent, build_ExpertIteration_Agent
from regym.rl_algorithms import rockAgent, paperAgent, scissorsAgent, randomAgent


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


def initialize_training_schemes(training_schemes_configs, task):
    '''
    Creates a list containing pointers to the relevant self_play training scheme functions
    :param candidate_training_schemes: requested training schemes
    :return: list containing pointers to the corresponding self_play training schemes functions
    '''
    def partial_match_build_function(self_play_name, config, task):
        if self_play_name.startswith('psro'):
            return PSRONashResponse(task=task, **config)
        if self_play_name.startswith('naiveselfplay'):
            return NaiveSelfPlay
        if self_play_name.startswith('deltalimituniform'):
            return DeltaLimitDistributionalSelfPlay(
                delta=config['delta'],
                distribution=np.random.choice,
                save_every_n_episodes=config.get('save_every_n_episodes', 1))
        if self_play_name.startswith('deltauniform'):
            return DeltaDistributionalSelfPlay(
                delta=config['delta'],
                distribution=np.random.choice,
                save_every_n_episodes=config.get('save_every_n_episodes', 1))
        else: raise ValueError(f'Unkown Self Play training scheme: {self_play_name}')
    return [partial_match_build_function(t_s.lower(), config, task) for t_s, config in training_schemes_configs.items()]


def initialize_agents(task, agent_configurations):
    '''
    Builds an agent for each agent in :param: agent_configurations
    suitable to act and process experience from :param: environment
    :param task: Task used to initialize agents on.
    :param agent_configurations: configuration dictionaries for each requested agent
    :returns: array of agents built according to their corresponding configuration dictionaries
    '''
    def partial_match_build_function(agent_name, task, config):
        if agent_name.startswith('tabularqlearning'): return build_TabularQ_Agent(task, config, agent_name)
        if agent_name.startswith('deepqlearning'): return build_DQN_Agent(task, config, agent_name)
        if agent_name.startswith('ppo'): return build_PPO_Agent(task, config, agent_name)
        if agent_name.startswith('expert_iteration'): return build_ExpertIteration_Agent(task, config, agent_name)
        else: raise ValueError('Unkown agent name: {agent_name}'.format(agent_name))
    return [partial_match_build_function(agent, task, config) for agent, config in agent_configurations.items()]


def initialize_fixed_agents(fixed_agents):
    '''
    Builds a fixed (stationary) agent for each agent in :param: fixed_agents.
    ASSUMPTION: Each agent is able to take actions in the environment that will be used for the experiment
    :param: List of requested fixed agent names to be created
    :return: array of initialized stationary agents
    '''
    fix_agent_build_functions = {'rockagent': rockAgent, 'paperagent': paperAgent, 'scissorsagent': scissorsAgent, 'randomagent': randomAgent}
    check_for_unknown_candidate_input(fix_agent_build_functions.keys(), fixed_agents, 'fixed_agents')
    return [fix_agent_build_functions[agent.lower()] for agent in fixed_agents]


def filter_relevant_configurations(experiment_config: Dict,
                                   target_configs: Dict[str, Dict], target_key: str):
    '''
    The config file allows to have configuration for RL algorithms that will not be used.
    This allows to keep all configuration in a single file.
    The configuration that will be used is explicitly captured in :param: experiment_config
    :param experiment_config: TODO
    :param target_configs: TODO
    :param key: TODO
    '''
    return {key: config for key, config in target_configs.items()
            if any(map(lambda elem: key.startswith(elem), experiment_config[target_key]))}
