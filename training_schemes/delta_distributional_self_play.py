import math

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
    subset_of_considered_agents = slice(math.ceil(delta * len(menagerie)), len(menagerie))
    valid_agents = menagerie[subset_of_considered_agents] + [training_agent]
    return map( lambda AgentHook : AgentHook(training=False, use_cuda=True), [distribution(valid_agents)] )


def curator(menagerie, training_agent, episode_trajectory):
    '''
    # TODO : save the agents on disk into MenagireInterface objects.
    
    :param menagerie: archive of agents selected by the curator and the potential opponents
    :param training_agent: AgentHook of the Agent currently being trained
    :returns: menagerie to be used in the next training episode.
    '''

    return menagerie + [traning_agent]
