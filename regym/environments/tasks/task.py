from typing import List, Tuple, Callable, Any, Dict, Optional, Union
from functools import reduce
from copy import deepcopy, copy
from enum import Enum
from multiprocessing import cpu_count
from dataclasses import dataclass, field

import gym
from torch.utils.tensorboard import SummaryWriter

import regym
from .regym_worker import RegymAsyncVectorEnv

from ..env_type import EnvType


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
    name: str                            # Environment name
    env: gym.Env                         # Underlying environment
    env_type: EnvType                    # Whether single agent or multiagent
    state_space_size: Optional[int]      # Total number of steps
    action_space_size: int               # Total number of actions

    observation_dim: Any                 # Dimension / shape of env observation
    observation_size: int                # Flattened size of env observations
    observation_type: str                # Whether continuous or discrete

    action_dim: Any                      # Dimension / shape of env action space
    action_size: int                     # Flattened size of action space
    action_type: str                     # Whether continuous or discrete

    num_agents: int                      # Num of agents to act in underlying env
    hash_function: Callable[[Any], int]  # Function mapping Observation to a
                                         # single integer. Required to use
                                         # tabular methods.

    wrappers: List[gym.Wrapper] = field(default_factory=list)

    # Properties accessed post initializer
    extended_agents: Dict = field(default_factory=dict)
    total_episodes_run: int = 0
    total_timesteps_run: int = 0

    def run_episode(self, agent_vector: List['Agent'], training: bool,
                    render_mode: Optional[str] = None) -> 'Trajectory':
        '''
        Runs an episode of the Task's underlying environment using the
        :param: agent_vector to populate the agents in the environment.
        If the flag :param: training is set, the agents in :param: agent_vector
        will be fed the 'experiences'* collected during the episode. These episodes
        can be visualized via :param: render_mode (where implemented :P)

        Depending on the Task.env_type, a different mathematical model
        is used to simulate an episode on the environment.

        *The term 'experience' is defined in regym.rl_algorithms.agents.Agent

        :param agent_vector: List of agents that will act in the environment,
                             this parameter will be extended with
                             `task.extended_agents` vector.
        :param training:     Whether to propagate experiences to agents.
                             Note that agents must also have their own
                             `Agent.training` flag set.
        :param render_mode:  String identifier representing what how to render the
                             of environment. It is up to the the underlying
                             `Task.env` to support different render_mode(s),
                             and it is not Regym's responsability.
        :returns: List of trajectories experienced by agents in
                  :param: agent_vector
        '''
        self._check_required_number_of_agents_are_present(len(agent_vector))
        extended_agent_vector = self._extend_agent_vector(agent_vector)
        if self.env_type == regym.environments.EnvType.SINGLE_AGENT:
            ts = regym.rl_loops.singleagent_loops.rl_loop.run_episode(self.env, extended_agent_vector[0], training, render_mode)
        if self.env_type == regym.environments.EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
            ts = regym.rl_loops.multiagent_loops.simultaneous_action_rl_loop.run_episode(self.env, extended_agent_vector, training, render_mode)
        if self.env_type == regym.environments.EnvType.MULTIAGENT_SEQUENTIAL_ACTION:
            ts = regym.rl_loops.multiagent_loops.sequential_action_rl_loop.run_episode(self.env, extended_agent_vector, training, render_mode)
        self.total_episodes_run += 1
        self.total_timesteps_run += len(ts)
        return ts

    def run_episodes(self,
                     agent_vector: List['Agent'],
                     num_episodes: int,
                     num_envs: int,
                     training: bool,
                     initial_episode: int = -1,
                     show_progress: bool = False,
                     summary_writer: Optional[Union[SummaryWriter, str]] = None) \
                     -> List['Trajectory']:
        '''
        Runs :param: num_episodes inside of the Task's underlying environment
        in :param: num_envs parallel environments.  If the flag
        :param: training is set, the agents in :param: agent_vector will be fed
        the 'experiences'* collected during the :param: num_episodes.

        Uses `RegymAsyncVectorEnv` under the hood.

        NOTES:
            - Because episodes are run in parallel until :param num_episodes:
              are finished, but agents are fed experiences (potentially) on
              every timestep, agents can experience trajectories which are not
              finished
            - Because more than one trajectory can finish at the same time,
              the number of returned trajectories is upper bounded by:
              (:param: num_episodes + :param: num_envs - 1)
            - Somehow calling this is slower than using `Task.run_episode`
              :param: num_episodes number of times... TODO: figure out where
              the slowness is coming from!

        :param agent_vector: List of agents that will act in the environment,
                             this parameter will be extended with
                             `task.extended_agents` vector.
        :param num_episodes: Target number of episodes to run task.env for.
        :param training:     Whether to propagate experiences to agents.
                             Note that agents must also have their own
                             `Agent.training` flag set.
        :param initial_episode: Episode ID used, used internally for
                                :param: show_progress and :param: summary_writter.
                                The default -1 indicates that self.total_episodes_run
                                will be used.
        :param show_progress: Whether to output a progress bar to stdout
        :param summary_writer: Summary writer to which log various metrics
        :returns: List of trajectories experienced by agents in
                  :param: agent_vector
        '''
        self._check_required_number_of_agents_are_present(len(agent_vector))
        extended_agent_vector = self._extend_agent_vector(agent_vector)

        if summary_writer:
            if isinstance(summary_writer, str):
                summary_writer = SummaryWriter(summary_writer)

        self.let_agents_access_each_other(
            agent_vector,
            num_envs if num_envs != -1 else cpu_count()
        )

        if initial_episode == -1: initial_episode = self.total_episodes_run

        for agent in agent_vector: agent.num_actors = num_envs

        self.start_agent_servers(agent_vector, num_envs)

        vector_env = RegymAsyncVectorEnv(self.name, num_envs, self.wrappers)
        trajectories = self.parallel_generate_trajectories(
            vector_env,
            extended_agent_vector,
            num_episodes,
            training,
            show_progress,
            summary_writer,
            initial_episode=initial_episode
        )

        self.total_episodes_run += num_episodes
        self.total_timesteps_run += sum(map(lambda t: len(t), trajectories))

        self.end_agent_servers(agent_vector)
        for agent in agent_vector: agent.reset_after_episodes()
        return trajectories


    def let_agents_access_each_other(self, agent_vector: List['Agent'], num_envs: int):
        '''
        Allows all agents in :param: agent_vector that need to access
        other agents, to access them. This is useful for instance in
        ExpertIterationAgent, where we can have a neural_net_server
        hosting a neural_nets from other agents

        :param agent_vector: Agents that are about to play in :param: self
        '''
        for i, agent in enumerate(agent_vector):
            if not agent.requires_acess_to_other_agents: continue
            other_agents = copy(agent_vector); other_agents.pop(i)
            agent.access_other_agents(other_agents, self, num_envs)


    def parallel_generate_trajectories(self, vector_env: RegymAsyncVectorEnv,
                                       agent_vector: List['Agent'],
                                       num_episodes: int,
                                       training: bool,
                                       show_progress: bool,
                                       summary_writer: Optional[SummaryWriter],
                                       initial_episode: int) -> List['Trajectory']:
        if self.env_type == EnvType.SINGLE_AGENT:
            ts = regym.rl_loops.singleagent_loops.rl_loop.async_run_episode(
                    vector_env, agent_vector[0], training, num_episodes)
        elif self.env_type == EnvType.MULTIAGENT_SEQUENTIAL_ACTION:
            ts = regym.rl_loops.multiagent_loops.vectorenv_sequential_action_rl_loop.async_run_episode(
                    vector_env, agent_vector, training, num_episodes, show_progress,
                    summary_writer, initial_episode)
        elif self.env_type == EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
            raise NotImplementedError('Simultaenous environments do not currently allow multiple environments. use Task.run_episode')
        return ts

    def extend_task(self, agents: Dict[int, 'Agent'], force: bool = False):
        '''
        A task is "extended" by preinserting agents into certain task positions
        before calling `Task.run_episode()` or `Task.run_episodes()`.
        An `N`-agent multiagent task can be extended with up to `N - 1` agents.
        Useful, for instance, to extend a task with a certain fixed agent(s)
        to benchmark against it.

        Trying to extend a task with agents on positions that have already been
        filled by previous calls to `Task.extend_task()` will raise an
        exception. Use :param: force.

        Only valid for multigent tasks.

        :params agents: Dict of [agent_position, agent].
        :param force: Boolen that allows to override agent positions which have
                      already been extended.
        '''
        if self.env_type == EnvType.SINGLE_AGENT:
            raise ValueError('SINGLE_AGENT tasks cannot be extended')
        for i, agent in agents.items():
            if i in self.extended_agents and not force:
                raise ValueError(f'Trying to overwrite agent {i}: {agent.name}. If sure, set param `force`.')
            self.extended_agents[i] = agent

    def _extend_agent_vector(self, agent_vector: List) -> List['Agent']:
        '''
        Extends :param: agent_vector with agents collected from previous
        calls to `Task.extend_task()`, before task `self` is run
        for some episodes.

        :param agent_vector: (Potentially partially complete) list of agents
                             which will act in this task.
        :returns: Complete list of agents which will act in task.
        '''
        agent_index = 0
        extended_agent_vector = []
        for i in range(self.num_agents):
            if i in self.extended_agents:
                extended_agent_vector.append(self.extended_agents[i])
            else:
                extended_agent_vector.append(agent_vector[agent_index])
                agent_index += 1

        return extended_agent_vector

    def _check_required_number_of_agents_are_present(self,
                                                     num_provided_agents: int):
        ''' Checks whether the task has enough agents to run episodes with '''
        if len(self.extended_agents) + num_provided_agents < self.num_agents:
            raise ValueError(f'Task {self.name} requires {self.num_agents} agents, but only {len(agent_vector)} agents were given (in :param agent_vector:). With {len(self.extended_agents)} currently pre-extended. See documentation for function Task.extend_task()')

    def start_agent_servers(self, agent_vector: List['Agent'], num_envs: int):
        '''
        Flags to all agents that will act in the task and require a server
        to start it based on their own logic.

        :param num_envs: Number of environments to be run simultaneously
        :param agent_vector: Agents that will act in task
        '''
        for agent in agent_vector:
            if agent.multi_action_requires_server:
                agent.start_server(num_connections=num_envs)

    def end_agent_servers(self, agent_vector: List['Agent']):
        '''
        Flags to all agents that acted in the task to shut down their servers.

        :param agent_vector: Agents that have acted in task
        '''
        for agent in agent_vector:
            if agent.multi_action_requires_server: agent.close_server()

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
        s = (f'Task: {self.name}\n'
             f'env: {self.env}\n'
             f'env_type: {self.env_type}\n'
             f'Env wrappers: {self.wrappers}\n'
             f'num_agents: {self.num_agents}\n'
             f'Extended_agents: {self.extended_agents}\n'
             f'observation_dim: {self.observation_dim}\n'
             f'observation_size: {self.observation_size}\n'
             f'observation_type: {self.observation_type}\n'
             f'state_space_size: {self.state_space_size}\n'
             f'action_space_size: {self.action_space_size}\n'
             f'action_dim: {self.action_dim}\n'
             f'action_type: {self.action_type}\n'
             f'hash_function: {self.hash_function}\n'
             f'total_episodes_run: {self.total_episodes_run}\n'
             f'total_timesteps_run: {self.total_timesteps_run}\n'
             )
        return s
