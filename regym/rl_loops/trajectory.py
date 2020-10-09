from typing import List, Dict, Any, Union
from dataclasses import dataclass, field

from regym.environments import EnvType


@dataclass
class Timestep:

    observation: Any
    action: Union[Any, List[Any]]
    reward: Union[float, List[float]]
    succ_observation: Any
    done: bool
    acting_agents: List[int]  # List of agents who acted at this timestep
    extra_info: Dict
    t: int


class Trajectory:

    def __init__(self, env_type: EnvType):
        self.env_type = env_type
        self._timesteps: List[Timestep] = []

    def add_timestep(self, o, a, r, succ_o, done,
                     acting_agents: List[int] = [0],
                     extra_info={}):
        '''
        TODO:
        '''
        if self.env_type == EnvType.SINGLE_AGENT: acting_agents = [0]
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

    def __getitem__(self, n):
        return self._timesteps[n]

    def __len__(self):
        return len(self._timesteps)

    def __iter__(self):
        return iter(self._timesteps)

    def __repr__(self):
        return f'{self.env_type} Trajectory. Length: {len(self._timesteps)}'
