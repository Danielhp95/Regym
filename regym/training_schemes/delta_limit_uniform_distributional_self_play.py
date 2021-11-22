from typing import Callable
import math

from ..rl_algorithms import AgentHook
from .delta_distributional_self_play import DeltaDistributionalSelfPlay


'''
Delta-limit-uniform distribution:
enforces a uniform opponent sampling, asymptotically,
without biasing towards earlier policies.
'''
class DeltaLimitDistributionalSelfPlay(DeltaDistributionalSelfPlay):

    def __init__(self, delta: float, distribution: Callable,
                 save_every_n_episodes: int = 1):
        super().__init__(delta, distribution, save_every_n_episodes)
        # The name is set in super().__init__
        # So setting a different name must happen _afterwards_
        self.name = f'limit-d={delta},{distribution.__name__}'

    def opponent_sampling_distribution(self, menagerie, training_agent):
        '''
        :param menagerie: archive of agents selected by the curator and the potential opponents
        :param training_agent: Agent currently being trained
        :returns: Agent, sampled from the menagerie, to be used as an opponent in the next episode
        '''
        latest_training_agent_hook = AgentHook(training_agent.clone(training=False))
        indices = range(len(menagerie) + 1) # +1 accounts for the training agent, not (yet) included in menagerie
        subset_of_considered_indices = slice(math.ceil(self.delta * len(menagerie)), len(indices))
        valid_agents_indices = indices[subset_of_considered_indices]
        n = len(valid_agents_indices)
        unormalized_ps = [1.0/((n * (n-i)**2)) for i in range(n)]
        sum_ps = sum(unormalized_ps)
        normalized_ps = [p / sum_ps for p in unormalized_ps]
        samples_indices = [self.distribution(valid_agents_indices, p=normalized_ps)]
        samples = [menagerie[i] if i < len(menagerie) else latest_training_agent_hook for i in samples_indices]
        return [AgentHook.unhook(sampled_hook_agent) for sampled_hook_agent in samples]

    def __repr__(self):
        return f'DeltaLimitDistributionalSelfPlay: delta={self.delta}, distribution={self.distribution}, save every n episodes: {self.save_every_n_episodes}'
