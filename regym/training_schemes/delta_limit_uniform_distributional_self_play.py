import math
from ..rl_algorithms import AgentHook

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
    indices = range(len(menagerie) + 1) # +1 accounts for the training agent, not (yet) included in menagerie
    subset_of_considered_indices = slice(math.ceil(delta * len(menagerie)), len(indices))
    valid_agents_indices = indices[subset_of_considered_indices]
    n = len(valid_agents_indices)
    unormalized_ps = [1.0/((n * (n-i)**2)) for i in range(n)]
    sum_ps = sum(unormalized_ps)
    normalized_ps = [p / sum_ps for p in unormalized_ps]
    samples_indices = [distribution(valid_agents_indices, p=normalized_ps)]
    samples = [menagerie[i] if i < len(menagerie) else latest_training_agent_hook for i in samples_indices]
    return [AgentHook.unhook(sampled_hook_agent) for sampled_hook_agent in samples]


def curator(menagerie, training_agent, episode_trajectory,
            training_agent_index, candidate_save_path):
    '''

    :param menagerie: archive of agents selected by the curator and the potential opponents
    :param training_agent: AgentHook of the Agent currently being trained
    :returns: menagerie to be used in the next training episode.
    '''

    return menagerie + [AgentHook(training_agent.clone(training=False), save_path=candidate_save_path)]
