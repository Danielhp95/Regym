from typing import List, Callable
import math
from copy import deepcopy

import torch

from regym.util import are_neural_nets_equal

'''
Based on the paper: Emergent Complexity via Multi-agent Competition
The implementation differs from Bansal's because this class
allows to use distributions which aren't uniform.
'''
class DeltaDistributionalSelfPlay():


    def __init__(self, delta: float, distribution: Callable,
                 save_every_n_episodes: int = 1,
                 save_after_policy_update: bool = True):
        '''
        :param delta: determines the percentage of the menagerie that will be
                      considered by the opponent_sampling_distribution:
                      - delta = 0 (all history),
                      - delta = 1 (only latest agent, Naive Self Play)
        :param distribution: Distribution to be used over the filtered set of agents.
        :param save_every_n_episodes: Number of episodes to skip before saving / adding an
                                      agent into the menagerie. Default value of 0 saves an
                                      agent every episode, which is very memory consuming.
        :param save_after_policy_update: If true, agents will only enter the menagerie if there
                                         is a change in the agent's neural net paramaters.
                                         If true, :param: save_every_n_episodes will be ignored
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

        self.save_after_policy_update = save_after_policy_update

    def opponent_sampling_distribution(self, menagerie: List['Agent'], training_agent: 'Agent') -> 'Agent':
        '''
        :param menagerie: archive of agents selected by the curator and the potential opponents
        :param training_agent: Agent currently being trained
        :param delta: determines the percentage of the menagerie that will be considered by the opponent_sampling_distribution. delta = 0 (all history), delta = 1 (only latest agent)
        :param distribution: Distribution to be used over the filtered set of agents.
        :returns: Agent, sampled from the menagerie, to be used as an opponent in the next episode
        '''
        latest_training_agent = training_agent.clone(training=False)
        indices = range(len(menagerie) + 1) # +1 accounts for the training agent, not (yet) included in menagerie
        subset_of_considered_indices = slice(math.ceil(self.delta * len(menagerie)), len(indices))
        valid_agents_indices = indices[subset_of_considered_indices]
        samples_indices = [self.distribution(valid_agents_indices)]
        samples = [menagerie[i] if i < len(menagerie) else latest_training_agent
                   for i in samples_indices]
        return [sampled_hook_agent for sampled_hook_agent in samples]

    def curator(self,
                menagerie: List['Agent'],
                training_agent: 'Agent',
                episode_trajectory: 'Trajectory',
                training_agent_index: int,
                candidate_save_path: str) -> List['Agent']:
        '''
        :param menagerie: archive of agents selected by the curator and the potential opponents
        :param training_agent: AgentHook of the Agent currently being trained
        :param episode_trajectory: Trajectory of the last episode where :param: training_agent trained
        :param training_agent_index: Index of :param: training_agent inside of the agent vector of
                                     the environment that produced :param: episode_trajectory
        :param candidate_save_path: Path where :param: training_agent will be stored if it's eligible
        :returns: menagerie to be used in the next training episode.
        '''

        menagerie_addition = []
        if self.save_after_policy_update:
            assert hasattr(training_agent, 'algorithm'), 'Saving after policy update only supported if policy is stored in Agent.algorithm.model'
            assert hasattr(training_agent.algorithm, 'model'), 'Saving after policy update only supported if policy is stored in Agent.algorithm.model'
            assert isinstance(training_agent.algorithm.model, torch.nn.Module)
            if len(menagerie) == 0 or not(are_neural_nets_equal(
                    menagerie[-1].algorithm.model, training_agent.algorithm.model)):
                print(training_agent.algorithm.num_updates)
                menagerie_addition = self.clone_agent_to_add_to_menagerie(training_agent, candidate_save_path)

        elif (self.save_skips_i % self.save_every_n_episodes) == 0:
            # By having a deep copy we are slowing things down, because
            # We could have a clone function. But at the pace that we change
            # the Agent class, this is not really worth it.
            menagerie_addition = self.clone_agent_to_add_to_menagerie(training_agent, candidate_save_path)
            self.save_skips_i += 1
        return menagerie + menagerie_addition

    def clone_agent_to_add_to_menagerie(self, training_agent: 'Agent', candidate_save_path: str) -> List['Agent']:
        '''
        Saves :param: agent in :param: candidate_save_path and returns a
        singleton list with a cloned version of :param: agent to add
        to be added to the menagerie
        '''
        cloned_agent = deepcopy(training_agent)
        cloned_agent.training = False
        torch.save(cloned_agent, candidate_save_path)
        return [cloned_agent]

    def __repr__(self):
        return f'DeltaDistributionalSelfPlay: delta={self.delta}, distribution={self.distribution}, save every n episodes: {self.save_every_n_episodes}'
