from copy import deepcopy
from enum import Enum
from typing import List, Callable, Any, Dict
from dataclasses import dataclass, field

import gym

import regym


class EnvType(Enum):
    '''
    Enumerator representing what kind of environment a task will deal with.
    Useful because different environments (simulatenous vs sequential) require
    a different underlying mathematical construct to simulate an episode
    '''
    SINGLE_AGENT = 'single-agent'
    MULTIAGENT_SIMULTANEOUS_ACTION = 'multiagent-simultaneous'
    MULTIAGENT_SEQUENTIAL_ACTION = 'multiagent-sequential'


@dataclass
class Task:
    r'''
    A Task is a thin layer of abstraction over OpenAI gym environments and
    Unity ML-agents executables, used across Regym.
    The main uses of Tasks are:
        - Initialize agents capable of acting in an environment via the
          `build_X_Agent()` functions where `X` is an algorithm from
          `regym.rl_algorithms`.
        - Run episodes of the underlying environment via the `task.run_episode` function.

    NOTE: Unless you know what you are doing, a Task should be generated thus:
    >>> from regym.environments import generate_task
    >>> task = generate_task('OpenAIGymEnv-v0')

    Tasks can encapsulate 3 types of environments. Captured in the class
    `regym.environments.EnvType`:
        - SINGLE_AGENT
        - MULTIAGENT_SIMULTANEOUS_ACTION
        - MULTIAGENT_SEQUENTIAL_ACTION

    Single agent environments are self-explanatory. In sequential action
    environments, the environment will process a single agent action on every
    `env.step` function call. Simultaenous action environments will take an
    action from every player on each `env.step` function call.

    For multiagent environments, it is mandatory to specify whether the
    actions are consumed simultaneously or sequentially by the environment.
    This is done via passing an EnvType to the `generate_task` function.

    >>> from regym.environments import EnvType
    >>> simultaneous_task = generate_task('SimultaneousEnv-v0', EnvType.MULTIAGENT_SIMULTANEOUS_ACTION)
    >>> sequential_task   = generate_task('SequentialsEnv-v0',  EnvType.MULTIAGENT_SEQUENTIAL_ACTION)
    '''

    # Properties set on __init__
    name: str
    env: gym.Env
    env_type: EnvType
    state_space_size: int
    action_space_size: int
    observation_dim: int
    observation_type: str
    action_dim: int
    action_type: str
    num_agents: int
    hash_function: Callable[[Any], int]

    # Properties accessed post initializer
    extended_agents: Dict = field(default_factory=dict)
    total_episodes_run: int = 0

    def extend_task(self, agents: Dict, force=False):
        ''' TODO: DOCUMENT, TEST '''
        if self.env_type == EnvType.SINGLE_AGENT:
            raise ValueError('SINGLE_AGENT tasks cannot be extended')
        for i, agent in agents.items():
            # Maybe refactor whole scope into a list comprehension
            if i in self.extended_agents and not force:
                raise ValueError(f'Trying to overwrite agent {i}: {agent.name}. If sure, set param `force`.')
            self.extended_agents[i] = agent

    def run_episode(self, agent_vector: List, training: bool, render_mode: str = ''):
        '''
        Runs an episode of the Task's underlying environment using the
        :param: agent_vector to populate the agents in the environment.
        If the flag :param: training is set, the agents in :param: agent_vector
        will be fed the 'experiences'* collected during the episode.

        Depending on the Task.env_type, a different mathematical model
        is used to simulate an episode an episode on the environment.

        *The term 'experience' is defined in regym.rl_algorithms.agents.Agent
        '''
        if len(self.extended_agents) + len(agent_vector) < self.num_agents:
            raise ValueError(f'Task {self.name} requires {self.num_agents} agents, but only {len(agent_vector)} agents were given (in :param agent_vector:). With {len(self.extended_agents)} currently pre-extended. See documentation for function Task.extend_task()')
        extended_agent_vector = self._extend_agent_vector(agent_vector)
        self.total_episodes_run += 1
        if self.env_type == EnvType.SINGLE_AGENT:
            return regym.rl_loops.singleagent_loops.rl_loop.run_episode(self.env, extended_agent_vector[0], training, render_mode)
        if self.env_type == EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
            return regym.rl_loops.multiagent_loops.simultaneous_action_rl_loop.run_episode(self.env, extended_agent_vector, training, render_mode)
        if self.env_type == EnvType.MULTIAGENT_SEQUENTIAL_ACTION:
            return regym.rl_loops.multiagent_loops.sequential_action_rl_loop.run_episode(self.env, extended_agent_vector, training, render_mode)

    def _extend_agent_vector(self, agent_vector: List):
        # This should be much prettier
        agent_index = 0
        extended_agent_vector = []
        for i in range(self.num_agents):
            if i in self.extended_agents:
                extended_agent_vector.append(self.extended_agents[i])
            else:
                extended_agent_vector.append(agent_vector[agent_index])
                agent_index += 1

        return extended_agent_vector

    def clone(self):
        cloned = Task(
                name=self.name,
                env=deepcopy(self.env),
                env_type=self.env_type,
                state_space_size=self.state_space_size,
                action_space_size=self.action_space_size,
                observation_dim=self.observation_dim,
                observation_type=self.observation_type,
                action_dim=self.action_dim,
                action_type=self.action_type,
                num_agents=self.num_agents,
                hash_function=self.hash_function)
        cloned.extended_agents = {k: agent.clone()
                                  for k, agent in self.extended_agents}
        cloned.total_episodes_run = self.total_episodes_run
        return cloned

    def __repr__(self):
        s = \
f'''
Task: {self.name}
env: {self.env}
env_type: {self.env_type}
num_agents: {self.num_agents}
Extended_agents: {self.extended_agents}
observation_dim: {self.observation_dim}
observation_type: {self.observation_type}
state_space_size: {self.state_space_size}
action_space_size: {self.action_space_size}
action_dim: {self.action_dim}
action_type: {self.action_type}
hash_function: {self.hash_function}
'''
        return s
