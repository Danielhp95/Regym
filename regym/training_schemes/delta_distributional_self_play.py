from typing import List, Callable
import math

from ..rl_algorithms import AgentHook

'''
Based on the paper: Emergent Complexity in Multi TODO
The implementation differs from Bansal's because this class
allows to use distributions which aren't uniform.
'''
class DeltaDistributionalSelfPlay():


    def __init__(self, delta: float, distribution: Callable,
                 save_every_n_episodes: int = 1):
        '''
        :param delta: determines the percentage of the menagerie that will be
                      considered by the opponent_sampling_distribution:
                      - delta = 0 (all history),
                      - delta = 1 (only latest agent, Naive Self Play)
        :param distribution: Distribution to be used over the filtered set of agents.
        :param save_every_n_episodes: Number of episodes to skip before saving / adding an
                                      agent into the menagerie. Default value of 0 saves an
                                      agent every episode, which is very memory consuming.
        '''
        if not (0 <= delta <= 1):
            raise ValueError(f':param: delta should be bound between [0, 1]. Given {delta}')
        if (save_every_n_episodes < 1):
            raise ValueError(f':param: save_every_n_episodes should be equal or greater than 1. Given {save_every_n_episodes}')

        self.name = f'd={delta},{distribution.__name__}'
        self.delta = delta
        self.distribution = distribution

        self.save_skips_i = 0
        self.save_every_n_episodes = save_every_n_episodes

    def opponent_sampling_distribution(self, menagerie: List['Agent'], training_agent: 'Agent') -> 'Agent':
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

        if (self.save_skips_i % self.save_every_n_episodes) == 0:
            menagerie_addition = [AgentHook(training_agent.clone(training=False),
                                            save_path=candidate_save_path)]
        else: menagerie_addition = []

        self.save_skips_i += 1
        return menagerie + menagerie_addition

    def __repr__(self):
        return f'DeltaDistributionalSelfPlay: delta={self.delta}, distribution={self.distribution}, save every n episodes: {self.save_every_n_episodes}'
