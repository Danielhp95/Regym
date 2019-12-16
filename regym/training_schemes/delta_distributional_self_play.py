import math
from ..rl_algorithms import AgentHook

'''
Based on the paper: Emergent Complexity in Multi TODO
The implementation differs from Bansal's because this class
allows to use distributions which aren't uniform.
'''
class DeltaDistributionalSelfPlay():


    def __init__(self, delta, distribution):
        '''
        :param delta: determines the percentage of the menagerie that will be
                      considered by the opponent_sampling_distribution:
                      - delta = 0 (all history),
                      - delta = 1 (only latest agent, Naive Self Play)
        :param distribution: Distribution to be used over the filtered set of agents.
        '''
        self.name = f'd={delta},{distribution.__name__}'
        self.delta = delta
        self.distribution = distribution

    def opponent_sampling_distribution(self, menagerie, training_agent):
        '''
        :param menagerie: archive of agents selected by the curator and the potential opponents
        :param training_agent: Agent currently being trained
        :param delta: determines the percentage of the menagerie that will be considered by the opponent_sampling_distribution. delta = 0 (all history), delta = 1 (only latest agent)
        :param distribution: Distribution to be used over the filtered set of agents.
        :returns: Agent, sampled from the menagerie, to be used as an opponent in the next episode
        '''
        latest_training_agent_hook = AgentHook(training_agent.clone(training=False))
        indices = range(len(menagerie) + 1) # +1 accounts for the training agent, not (yet) included in menagerie
        subset_of_considered_indices = slice(math.ceil(self.delta * len(menagerie)), len(indices))
        valid_agents_indices = indices[subset_of_considered_indices]
        samples_indices = [self.distribution(valid_agents_indices)]
        samples = [menagerie[i] if i < len(menagerie) else latest_training_agent_hook
                   for i in samples_indices]
        return [AgentHook.unhook(sampled_hook_agent) for sampled_hook_agent in samples]

    def curator(self, menagerie, training_agent, episode_trajectory, training_agent_index, candidate_save_path):
        '''
        :param menagerie: archive of agents selected by the curator and the potential opponents
        :param training_agent: AgentHook of the Agent currently being trained
        :returns: menagerie to be used in the next training episode.
        '''

        return menagerie + [AgentHook(training_agent.clone(training=False), save_path=candidate_save_path)]
