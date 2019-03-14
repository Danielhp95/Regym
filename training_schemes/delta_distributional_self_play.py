import os
import sys

sys.path.append(os.path.abspath('..'))

import math
from rl_algorithms import AgentHook

'''
Based on the paper: Emergent Complexity in Multi TODO
The implementation differs from Bansal's because this class
allows to use distributions which aren't uniform.
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
    return [AgentHook.unhook(sampled_hook_agent) for sampled_hook_agent in [distribution(valid_agents)]]


def curator(menagerie, training_agent, episode_trajectory, iteration):
    '''
    :param menagerie: archive of agents selected by the curator and the potential opponents
    :param training_agent: AgentHook of the Agent currently being trained
    :param episode_trajectory: trajectory of states followed by the training_agent.
    :param iteration: int, specifies the iteration of the training process at which the training_agent is yielded.
    :returns: menagerie to be used in the next training episode.
    '''
    return menagerie( training_agent, iteration)