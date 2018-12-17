'''
Classical notion of self-play. Where the opponent is ALWAYS the same as the policy that is being learnt.
'''


def opponent_sampling_distribution(menagerie, training_policy):
    '''
    :param menagerie: archive of policies selected by the curator and the potential opponents
    :param training_policy: Policy currently being trained
    :returns: Policy, sampled from the menagerie, to be used as an opponent in the next episode
    '''
    return [training_policy]


def curator(menagerie, training_policy, episode_trajectory):
    '''
    :param menagerie: archive of policies selected by the curator and the potential opponents
    :param training_policy: Policy currently being trained
    :returns: menagerie to be used in the next training episode.
    '''
    return menagerie
