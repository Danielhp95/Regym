from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass

import numpy as np

import regym


@dataclass
class Timestep:

    observation: Any
    action: Union[Any, List[Any]]
    reward: Union[float, List[float]]
    succ_observation: Any
    done: bool
    acting_agents: List[int]  # List of agents who acted at this timestep
    extra_info: Dict
    t: int  # Time at which this timestep was found in an environment


class Trajectory:

    def __init__(self, env_type: 'regym.environments.EnvType',
                 num_agents: int = 1):
        self.env_type = env_type
        self.num_agents = num_agents
        self._timesteps: List[Timestep] = []

    def add_timestep(self, o, a, r, succ_o, done,
                     acting_agents: Optional[List[int]] = None,
                     extra_info={}):
        if self.env_type == regym.environments.EnvType.SINGLE_AGENT: acting_agents = [0]
        t = Timestep(
            observation=o,
            action=a,
            reward=r,
            succ_observation=succ_o,
            done=done,
            acting_agents=acting_agents,
            extra_info=extra_info,
            t=len(self._timesteps))
        self._timesteps.append(t)

    @property
    def winner(self) -> int:
        '''
        Winner is defined as the agent with the largest cumulative reward.
        '''
        if len(self._timesteps) == 0 or not(self._timesteps[-1].done):
            raise AttributeError('Attempted to extract the winner of an unfinished trajectory')
        if self.env_type == regym.environments.EnvType.SINGLE_AGENT:
            raise AttributeError('A trajectory only has a winner in multigent environments')
        cumulative_reward_vector = self.cumulative_reward
        indexes_max_score = np.argwhere(cumulative_reward_vector == np.amax(cumulative_reward_vector))
        return np.random.choice(indexes_max_score.flatten().tolist())

    def agent_specific_cumulative_reward(self, agent_i: int) -> float:
        '''
        Cumulative reward for a specific agent
        '''
        if self.env_type == regym.environments.EnvType.SINGLE_AGENT:
            raise AttributeError('This function can only be called on multiagent trajectories')
        if not (0 <= agent_i < self.num_agents):
            raise ValueError(f':param: agent_i should be between 0 and {self.num_agents - 1}')
        return sum(map(lambda timestep: timestep.reward[agent_i],
                       self._timesteps))

    @property
    def cumulative_reward(self) -> Union[float, List[float]]:
        '''
        TODO: nicer documentation
        '''
        if self.env_type == regym.environments.EnvType.SINGLE_AGENT:
            cum_reward = sum(map(lambda experience: experience.reward,
                                 self._timesteps))
        else:
            cum_reward = [self.agent_specific_cumulative_reward(a_i)
                          for a_i in range(self.num_agents)]
        return cum_reward

    # Formerly known as trajectory reward
    def agent_reward(self, agent_position: int) -> float:
        return sum(map(lambda t: t.reward[agent_position], self._timesteps))

    def __getitem__(self, n):
        return self._timesteps[n]

    def __len__(self):
        return len(self._timesteps)

    def __iter__(self):
        return iter(self._timesteps)

    def __repr__(self):
        return f'Trajectory, {self.env_type}, num agents: {self.num_agents}. Length: {len(self._timesteps)}'
