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
    def t(self) -> int:
        ''' Current trajectory timestep '''
        return len(self._timesteps)

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

    def last_acting_timestep_for_agent(self, agent_i: int, skip=0) -> Timestep:
        '''
        Backwards search for a timestep where :param: agent_i last acted.
        Useful when constructing an "experience tuple" for an agent in
        sequential games.

        :param: skip is used to skip a fixed number of timesteps where the agent
        acts.

        '''
        if not (0 <= agent_i < self.num_agents):
            raise ValueError(f':param: agent_i should be between 0 and {self.num_agents - 1}')
        if not(0 <= skip < len(self._timesteps)):
            raise ValueError(f'Param :skip: must lay between [0, len(trajectory)]')
        remaining_skips = skip
        for t in reversed(self._timesteps):
            if agent_i in t.acting_agents:
                if remaining_skips <= 0:
                    return t
                remaining_skips -= 1
        raise ValueError(f'Could not find a timestep (with {skip} skips) steps back where agent ({agent_i}) acts')

    @property
    def observations(self) -> List:
        return [t.observation for t in self._timesteps]

    @property
    def actions(self) -> List:
        return [t.action for t in self._timesteps]

    @property
    def rewards(self) -> List:
        return [t.reward for t in self._timesteps]

    @property
    def succ_observations(self) -> List:
        return [t.succ_observation for t in self._timesteps]

    def __getitem__(self, n) -> Union[Timestep, List[Timestep]]:
        return self._timesteps[n]

    def __len__(self) -> int:
        return len(self._timesteps)

    def __iter__(self):
        return iter(self._timesteps)

    def __repr__(self):
        return f'Trajectory, {self.env_type}, num agents: {self.num_agents}. Length: {len(self._timesteps)}'


def compute_winrates(trajectories: List[Trajectory]) -> List[float]:
    '''
    Computes the winrates of all agents from :param: trajectories
    by extracting the winner of each trajectory in :param: trajectories

    ASSUMPTION: all trajectories feature the same number of agents
    :param trajectories: List of trajectories from which to extract winrates
    :returns: List containing winrate for each agent that acted in
              :param: trajectories. Entry 'i' corresponds to the winrate
              of the 'ith' agent.
    '''
    num_agents = trajectories[0].num_agents
    winners = list(map(lambda t: t.winner, trajectories))
    return [winners.count(a_i) / len(winners) for a_i in range(num_agents)]
