from typing import List
import numpy as np
import regym

from regym.environments import Task, EnvType
from regym.rl_algorithms.agents import Agent
from regym.util import play_multiple_matches


def benchmark_agents_on_tasks(tasks: List[Task],
                              agents: List[Agent],
                              num_episodes: int,
                              populate_all_agents=False) -> np.ndarray:
    '''
    Benchmark :param: agents in :param: tasks for :param: num_episodes.
    Tasks must either be ALL EnvType.SINGLE_AGENT
    or ALL multiagent variants.

    TODO: maybe if multiple agents are passed in a single agent task,
          they are all benchmarked on said task?

    Returns:
        If :param: tasks are SINGLE_AGENT:
            this function returns the reward obtained 
            by :param: agent in :param: tasks averaged over :param: num_episodes.
        If :param: tasks are multiagent,:
            this function returns the winrate of agent 1 (in the n-agent task)
            against all other agents.

    TODO: improve this function to be more useful for n-player games.

    :param tasks: Tasks where the :param: agent(s) will be benchmarked
    :param agents: Agents that will populate the environment
    :param num_episodes: Number of episodes used to compute benchmarking statistics
    :param populate_all_agents: If a single agent is provided in :param: agents,
                                this flag indicates whether that agent's policy
                                will populate all other agents spots in the environment.
    '''
    check_input_validity(tasks, agents, num_episodes, populate_all_agents)
    # TODO: for single agent tasks we can't pass a vector, change naming
    winrates = []
    for t in tasks:
        player_winrates = play_multiple_matches(task=t,
                                                agent_vector=agents,
                                                n_matches=num_episodes)
        winrates.append(player_winrates[0])
    return winrates


def check_input_validity(tasks: List[Task], agents: List[Agent],
                         num_episodes: int, populate_all_agents: bool):
    all_tasks_are_singleagent = all(map(lambda t: t.env_type == EnvType.SINGLE_AGENT, tasks))
    all_tasks_are_multiagent = all(map(lambda t: t.env_type != EnvType.SINGLE_AGENT, tasks))

    if not all_tasks_are_singleagent and not all_tasks_are_multiagent:
        raise ValueError('All tasks must either be EnvType.SINGLE_AGENT or multiagent EnvTypes')

    if all_tasks_are_singleagent and len(agents) != 1:
        raise NotImplementedError('Bencharmarking multiple agents on SINGLE_AGENT tasks is not yet supported')

    if all_tasks_are_multiagent and not populate_all_agents:
        all_tasks_have_same_agents = all(map(lambda t: t.num_agents == tasks[0].num_agents, tasks))
        if not all_tasks_have_same_agents:
            raise ValueError(f'All tasks must require the same number of agents')

        agents_required = (tasks[0].num_agents - len(tasks[0].extended_agents))
        if len(agents) != agents_required:
            raise ValueError(f'Tasks require {agents_required} agent(s) but {len(agents)} were provided. Consider using populate_all_agents param')

    if all_tasks_are_multiagent and populate_all_agents and len(agents) != 1:
        raise ValueError(f'Only 1 agent can be used to populate all other agents if flag `populate_all_agents` is set. {len(agents)} agents were given.')

    if num_episodes <= 0:
        raise ValueError(f'Param `num_episodes` was {num_episodes}. It must be >= 0')
