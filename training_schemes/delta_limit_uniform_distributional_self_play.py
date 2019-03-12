import os
import sys

sys.path.append(os.path.abspath('..'))

import math
from rl_algorithms import AgentHook

'''
Delta-limit-uniform distribution: 
enforces a uniform opponent sampling, asymptotically,
without biasing towards earlier policies.
'''


def opponent_sampling_distribution(menagerie, training_agent, delta, distribution):
    '''
    :param menagerie: archive of agents selected by the curator and the potential opponents
    :param training_agent: Agent currently being trained
    :param delta: determines the percentage of the menagerie that will be considered by the opponent_sampling_distribution. delta = 0 (all history), delta = 1 (only latest agent)
    :param distribution: Distribution to be used over the filtered set of agents.
    :returns: Agent, sampled from the menagerie, to be used as an opponent in the next episode
    '''
    latest_training_agent_hook = AgentHook(training_agent.clone(training=False))
    subset_of_considered_agents = slice(math.ceil(delta * len(menagerie)), len(menagerie))
    valid_agents = menagerie[subset_of_considered_agents] + [latest_training_agent_hook]
    n = len(valid_agents)
    unormalized_ps = [1.0/((n * (n-i)**2)) for i in range(n)]
    sum_ps = sum(unormalized_ps)
    normalized_ps = [p / sum_ps for p in unormalized_ps]
    return [AgentHook.unhook(sampled_hook_agent) for sampled_hook_agent in [distribution(valid_agents, p=normalized_ps)]]


def curator(menagerie, training_agent, episode_trajectory, candidate_save_path):
    '''

    :param menagerie: archive of agents selected by the curator and the potential opponents
    :param training_agent: AgentHook of the Agent currently being trained
    :returns: menagerie to be used in the next training episode.
    '''

    return menagerie + [AgentHook(training_agent.clone(training=False), save_path=candidate_save_path)]
