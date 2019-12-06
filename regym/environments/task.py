from enum import Enum
from typing import Callable, Any
import gym

from regym.rl_loops.singleagent_loops import rl_loop
from regym.rl_loops.multiagent_loops import sequential_action_rl_loop
from regym.rl_loops.multiagent_loops import simultaneous_action_rl_loop


class EnvType(Enum):
    SINGLE_AGENT = 'single-agent'
    MULTIAGENT_SIMULTANEOUS_ACTION = 'multiagent-simultaneous'
    MULTIAGENT_SEQUENTIAL_ACTION = 'multiagent-sequential'


class Task:

    def __init__(self, name: str,
                 env: gym.Env,
                 env_type: EnvType,
                 state_space_size: int,
                 action_space_size: int,
                 observation_dim: int,
                 observation_type: str,
                 action_dim: int,
                 action_type: str,
                 hash_function: Callable[[Any], int]):
        '''
        TODO Document
        '''
        self.name = name
        self.env = env
        self.env_type = env_type
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.observation_dim = observation_dim
        self.observation_type = observation_type
        self.action_dim = action_dim
        self.action_type = action_type
        self.hash_function = hash_function

        self.total_episodes_run = 0


    def run_episode(self, agent_vector, training):
        '''
        TODO
        '''
        self.total_episodes_run += 1
        if self.env_type == EnvType.SINGLE_AGENT:
            return rl_loop.run_episode(self.env, agent_vector, training)
        if self.env_type == EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
            return simultaneous_action_rl_loop.run_episode(self.env, agent_vector, training)
        if self.env_type == EnvType.MULTIAGENT_SEQUENTIAL_ACTION:
            return sequential_action_rl_loop.run_episode(self.env, agent_vector, training)

    def __repr__(self):
        s = \
f'''
Task: {self.name}
env: {self.env}
env_type: {self.env_type}
observation_dim: {self.observation_dim}
observation_type: {self.observation_type}
state_space_size: {self.state_space_size}
action_space_size: {self.action_space_size}
action_dim: {self.action_dim}
action_type: {self.action_type}
hash_function: {self.hash_function}
'''
        return s
