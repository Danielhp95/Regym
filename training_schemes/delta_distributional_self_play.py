import math

'''
Based on the paper: Emergent Complexity in Multi TODO
The implementation differs from Bansal's because this class
allows to use distributions which aren't uniform.
'''


def opponent_sampling_distribution(menagerie, training_policy, delta, distribution):
    '''
    :param menagerie: archive of policies selected by the curator and the potential opponents
    :param training_policy: Policy currently being trained
    :param delta: determines the percentage of the menagerie that will be considered by the opponent_sampling_distribution. delta = 0 (all history), delta = 1 (only latest policy)
    :param distribution: Distribution to be used over the filtered set of policies.
    :returns: Policy, sampled from the menagerie, to be used as an opponent in the next episode
    '''
    subset_of_considered_policies = slice(math.ceil(delta * len(menagerie)), len(menagerie))
    valid_policies = menagerie[subset_of_considered_policies] + [training_policy]
    return [distribution(valid_policies)]


def curator(menagerie, training_policy, episode_trajectory):
    '''
    :param menagerie: archive of policies selected by the curator and the potential opponents
    :param training_policy: Policy currently being trained
    :returns: menagerie to be used in the next training episode.
    '''
    return menagerie
