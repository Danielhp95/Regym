'''
Classical notion of self-play. Where the opponent is ALWAYS the same as the agent that is being learnt.
'''


def opponent_sampling_distribution(menagerie, training_agent):
    '''
    :param menagerie: archive of agents selected by the curator and the potential opponents
    :param training_agent: AgentHook of the agent that is currently being trained
    :returns: Agent, sampled from the menagerie, to be used as an opponent in the next episode
    '''
    return [training_agent.clone(training=False)]


def curator(menagerie, training_agent, episode_trajectory,
            training_agent_index, candidate_save_path):
    '''
    :param menagerie: archive of agents selected by the curator and the potential opponents
    :param training_agent: Agent currently being trained
    :returns: menagerie to be used in the next training episode.
    '''
    return menagerie
