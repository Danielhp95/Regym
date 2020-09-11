from functools import reduce
from typing import List, Tuple, Any, Dict, NewType
from copy import deepcopy

import numpy as np
import gym

import regym
from regym.rl_algorithms.agents import Agent
from regym.environments.tasks import RegymAsyncVectorEnv

from regym.rl_loops.utils import update_parallel_sequential_trajectories, update_finished_trajectories
from regym.rl_loops.utils import agents_to_update_finished_trajectory_sequential_env


def async_run_episode(env: RegymAsyncVectorEnv, agent_vector: List[Agent],
                      training: bool, num_episodes: int) \
                      -> List[List[Tuple[Any, Any, Any, Any, bool]]]:
    '''
    Runs :param: num_episodes of asynchronous environment :param: env
    with agents specified in :param: agent_vector.

    For model-free agents, observations are batched, to be easily
    managed by neural networks. For model-based agents, an array of environment
    copies is handed over to the agent.

    NOTES:
    - Currently the :param: env runs `env.num_envs` asynchronous environments.
    Because more than one trajectory can finish at the same time,
    this function can return a number of trajectories in the range:
     $[num_episodes, num_episodes + (env.num_envs - 1)]$
    - Because some environments will be in the middle of an episode when
      this function returns, those trajectories won't appear in the output
      of this function, even though they have been processed by agents in
      :param: agent_vector.

    :param env: RegymAsyncVectorEnv where agents will play
    :param agent_vector: Vector containing agent policies
    :param training: Whether to propagate experiences to agents
                     in :param: agent_vector
    :param num_episodes: Number of target episodes to run environment for
    :returns: List of environment trajectories experienced during simulation.
    '''
    ongoing_trajectories: List[List[Tuple[Any, Any, Any, Any, bool]]]
    ongoing_trajectories = [[] for _ in range(env.num_envs)]
    finished_trajectories = []

    obs = env.reset()
    current_players: List[int] = [0] * env.num_envs
    legal_actions: List[List] = [None] * env.num_envs # Revise
    num_agents = len(agent_vector)
    while len(finished_trajectories) < num_episodes:

        # Take action
        action_vector = multienv_choose_action(
                agent_vector, env, obs, current_players, legal_actions)

        # Environment step
        succ_obs, rewards, dones, infos = env.step(action_vector)

        # Update trajectories:
        update_parallel_sequential_trajectories(ongoing_trajectories,
                            action_vector, obs, rewards, succ_obs, dones)

        # Update agents
        if training: propagate_experiences(agent_vector, ongoing_trajectories)

        # Update observation
        obs = succ_obs

        # Update current players and legal actions
        legal_actions = [info.get('legal_actions', None) for info in infos]
        current_players = [info.get('current_player',
                                    (current_players[e_i] + 1) % num_agents)
                           for e_i, info in enumerate(infos)]

        # Deal with episode termination
        done_envs = [i for i in range(len(dones)) if dones[i]]
        if len(done_envs) > 0:
            finished_trajectories, ongoing_trajectories = \
                    handle_finished_episodes(training, agent_vector,
                            ongoing_trajectories, done_envs,
                            finished_trajectories, current_players)

    return finished_trajectories


def multienv_choose_action(agent_vector: List[Agent],
                           env: RegymAsyncVectorEnv, obs,
                           current_players: List[int],
                           legal_actions: Dict[int, List[int]]) -> List[int]:
    '''
    Choose an action for each environment in (multienv) :param: env from
    agents in :param: agent_vector, constrained by :param: legal_actions, as
    prescribed by :param: current_players.

    :param: observations and :param: legal_actions from multiple environments
    where the same agent is meant to act will be batched to a single
    `Agent.take_action` call to reduce computational overhead.

    :param agent_vector: Vector containing agent policies
    :param env: RegymAsyncVectorEnv where agents are acting
    :param obs: TODO
    :param current_players: List indicating which agent should act on
                            each environment
    :param legal_actions: Dict indicating which actions are allowed on each
                          environment
    :returns: Vector containing one action to be executed on each environment
    '''
    action_vector: List[int] = [None] * env.num_envs

    agent_signals = extract_signals_for_acting_agents(
            agent_vector, obs, current_players, legal_actions)

    for a_i, signals in agent_signals.items():
        a = agent_vector[a_i]
        partial_action_vector = compute_partial_action_vector(a, signals,
                                                              env, a_i)
        for env_id, action in zip(signals['env_ids'], partial_action_vector):
            assert action_vector[env_id] is None, 'Attempt to override an action'
            action_vector[env_id] = action
    return action_vector


def compute_partial_action_vector(agent: Agent,
                                  signals: Dict[str, Any],
                                  env: RegymAsyncVectorEnv,
                                  agent_index: int) -> List[int]:
    '''
    :param agent_vector: Vector containing agent policies
    :param signals: Environment signals per agent required to take actions
    :param env: RegymAsyncVectorEnv where agents are acting
    :returns: Actions to be taken by :param: agent
    '''
    if not agent.requires_environment_model:
        partial_action_vector = agent.model_free_take_action(
                signals['obs'], legal_actions=signals['legal_actions'],
                multi_action=True)
    else:
        envs = env.get_envs()
        relevant_envs = {e_i: envs[e_i] for e_i in signals['env_ids']}
        observations = {e_i: o for e_i, o in zip(signals['env_ids'], signals['obs'])}
        partial_action_vector = agent.model_based_take_action(
                relevant_envs, observations, agent_index, multi_action=True)
    return partial_action_vector


def handle_finished_episodes(training: bool, agent_vector: List[Agent],
                             ongoing_trajectories: List[List],
                             done_envs: List[int],
                             finished_trajectories: List[List],
                             current_players: List[int]) \
                             -> Tuple[List[List], List[List]]:
    if training:
        propagate_last_experiences(agent_vector, ongoing_trajectories,
                                   done_envs)
    # Reset players and trajectories
    # Why are we returning ongoing trajectories twice?
    ongoing_trajectories, finished_trajectories = update_finished_trajectories(
                    ongoing_trajectories, finished_trajectories, done_envs)
    current_players, ongoing_trajectories = reset_players_and_trajectories(
            done_envs, current_players, ongoing_trajectories)
    return finished_trajectories, ongoing_trajectories


def reset_players_and_trajectories(done_envs: List[int],
                                   current_players: Dict[int, int],
                                   ongoing_trajectories: List[List[Tuple]]) \
                                   -> Tuple[Dict[int, int], List[List[Tuple]]]:
    for i, e_i in enumerate(done_envs):
        current_players[e_i] = 0
        ongoing_trajectories[e_i] = []
    return current_players, ongoing_trajectories


def propagate_experiences(agent_vector: List[Agent], trajectories: List[List]):
    '''
    Batch propagates experiences from :param: trajectories to each
    corresponding agent in :param: agent_vector.

    ASSUMES that turns are taken in clockwise fashion:
        - Player 1 acts, player 2 acts..., player n acts, player 1 acts...
        - where n is the length of :param: agent_vector
    '''
    agent_to_update_per_env = {i: len(t) % len(agent_vector)
                               for i, t in enumerate(trajectories)
                               if len(t) >= len(agent_vector)}
    if agent_to_update_per_env == {}:  # No agents to update
        return

    agents_to_update = set(agent_to_update_per_env.values())
    environment_per_agents = {a_i: [env_i
                                    for env_i, a_j in agent_to_update_per_env.items()
                                    if a_i == a_j]
                              for a_i in agents_to_update}

    agent_experiences = collect_agent_experiences_from_trajectories(
            agents_to_update, agent_to_update_per_env, trajectories, agent_vector)

    propagate_batched_experiences(agent_experiences, agent_vector, environment_per_agents)


def propagate_batched_experiences(agent_experiences: Dict[int, List[Tuple]],
                                  agent_vector: List[Agent],
                                  environment_per_agents: Dict[int, List[int]]):
    for a_i, experiences in agent_experiences.items():
        if agent_vector[a_i].training:
            agent_vector[a_i].handle_multiple_experiences(
                    experiences, environment_per_agents[a_i])


def propagate_last_experiences(agent_vector: List['Agent'],
                               trajectories: List[List], done_envs: List[int]):
    ''' TODO '''

    agents_to_update_per_env = compute_agents_to_update_per_env(
            trajectories, done_envs, agent_vector) 

    agents_to_update = set(reduce(lambda acc, x: acc + x,
                                  agents_to_update_per_env.values(), []))
    environment_per_agents = {a_i: [env_i
                                    for env_i, agent_ids in agents_to_update_per_env.items()
                                    if a_i in agent_ids]
                              for a_i in agents_to_update}

    agent_experiences = {a_i: [] for a_i in agents_to_update}
    for a_i, envs in environment_per_agents.items():
        for e_i in envs:
            reward = trajectories[e_i][-1][2][a_i]  # Hardcoding reward index
            succ_obs = trajectories[e_i][-1][3][a_i]  # Hardcoding succ_obs index
            o, a = get_last_observation_and_action_for_agent(
                    a_i, trajectories[e_i], len(agent_vector))
            agent_experiences[a_i] += [(o, a, reward, succ_obs, True)]

    propagate_batched_experiences(agent_experiences, agent_vector, environment_per_agents)

def compute_agents_to_update_per_env(trajectories, done_envs, agent_vector):
    num_agents = len(agent_vector)
    agents_to_update_per_env = {
        done_e_i: agents_to_update_finished_trajectory_sequential_env(
            len(trajectories[done_e_i]), num_agents)
        for done_e_i in done_envs
        if len(trajectories[done_e_i]) >= len(agent_vector)}  # is this check necessary?
    return agents_to_update_per_env


def collect_agent_experiences_from_trajectories(agents_to_update: List[int],
                                                agent_to_update_per_env: Dict[int, int],
                                                trajectories: List[List],
                                                agent_vector: List[Agent]) \
                                                -> Dict[int, Any]:
    '''
    Collects the latests experience from :param: trajectories, for each
    :param: agents_to_update. Each agent collects experiences according to
    :param: agent_to_update_per_env.

    :param agents_to_update: List of all agents that need to be updated
    :param agent_to_update_per_env: Mapping from env_id to which agent_id
                                    needs to be updated
    :param trajectories: Trajectories for ongoing episodes
    :param agent_vector: List of agents acting in current environment
    '''
    agent_experiences = {a_i: [] for a_i in agents_to_update}

    for env_i, target_agent in agent_to_update_per_env.items():
        experience = extract_latest_experience(target_agent,
                           trajectories[env_i], agent_vector)
        agent_experiences[target_agent] += [experience]
    return agent_experiences


def extract_latest_experience(agent_id: int, trajectory: List, agent_vector: List):
    '''
    ASSUMPTION:
        - every non-terminal observation corresponds to
          the an information set unique for the player whose turn it is.
          This means that each "experience" is from which an RL agent will learn
          (o, a, r, o') is fragmented throughout the trajectory. This function
          "stiches together" the right environmental signals, ensuring that
          each agent only has access to information from their own information sets.

    :param agent_id: Index of agent which will receive a new experience
    :param trajectory: Current episode trajectory
    :param agent_vector: List of agents acting in current environment
    '''
    o, a = get_last_observation_and_action_for_agent(agent_id, trajectory,
                                                     len(agent_vector))
    (_, _, reward, succ_observation, done) = trajectory[-1]
    return (o, a, reward[agent_id], succ_observation[agent_id], done)


def extract_signals_for_acting_agents(agent_vector: List[Agent], obs,
                                      current_players: List[int],
                                      legal_actions) \
                                      -> Dict[int, Dict[str, List]]:
    '''
    Creates a dictionary contaning the mapping:
    - For each player in :param: current_players,
      a mapping to another dictionary containing:
    - (1) Environment observations, (2) legal actions on each env, (3) env_ids

    :param agent_vector:
    '''
    agent_signals: Dict[int, Dict[str, List]] = dict()

    # Extract signals for each acting agent
    for e_i, cp in enumerate(current_players):
        if cp not in agent_signals:
            agent_signals[cp] = dict()
            agent_signals[cp]['obs'] = []
            agent_signals[cp]['legal_actions'] = []
            agent_signals[cp]['env_ids'] = []
        agent_signals[cp]['obs'] += [obs[cp][e_i]]
        agent_signals[cp]['legal_actions'] += [legal_actions[e_i]]
        agent_signals[cp]['env_ids'] += [e_i]
    return agent_signals


def get_last_observation_and_action_for_agent(target_agent_id: int,
                                              trajectory: List, num_agents: int) -> Tuple:
    '''
    ASSUMES that turns are taken in clockwise fashion:
        - Player 1 acts, player 2 acts..., player n acts, player 1 acts...
        - where n is the length of :param: agent_vector

    Obtains the last observation and action for agent :param: target_agent_id
    from the :param: trajectory.

    :param target_agent_id: Index of agent whose last observation / action
                            we are searching for
    :param trajectory: Sequence of (o_i,a_i,r_i,o'_{i+1}) for all players i.
    :param num_agents: Number of agents acting in the current environment
    :returns: The last observation (information state) and action taken
              at such observation by player :param: target_agent_id.
    '''
    # Offsets are negative, exploiting Python's backwards index lookup
    previous_timestep = trajectory[-num_agents]
    last_observation = previous_timestep[0][target_agent_id]
    last_action = previous_timestep[1]
    return last_observation, last_action
