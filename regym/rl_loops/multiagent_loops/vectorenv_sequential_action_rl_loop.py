from functools import reduce
from typing import List, Tuple, Any, Dict, Optional
from copy import deepcopy
from time import time

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym

import regym
from regym.rl_algorithms.agents import Agent
from regym.environments.tasks import RegymAsyncVectorEnv
from regym.rl_loops.utils import update_parallel_sequential_trajectories, update_finished_trajectories
from regym.rl_loops.utils import agents_to_update_finished_trajectory_sequential_env
from regym.rl_loops.utils import extract_latest_experience_sequential_trajectory

from regym.rl_loops.trajectory import Trajectory


def async_run_episode(env: RegymAsyncVectorEnv,
                      agent_vector: List[Agent],
                      training: bool,
                      num_episodes: int,
                      show_progress: bool = False,
                      summary_writer: Optional[SummaryWriter] = None,
                      initial_episode: int = 0) \
                      -> List[Trajectory]:
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
    :param show_progress: Whether to output a progress bar to stdout
    :param summary_writer: Summary writer to which log various metrics
    :param initial_episode: Initial episode
    :returns: List of environment trajectories experienced during simulation.
    '''
    # Initialize trajectories
    ongoing_trajectories = [Trajectory(env_type=regym.environments.EnvType.MULTIAGENT_SEQUENTIAL_ACTION,
                                       num_agents=len(agent_vector))
                            for _ in range(env.num_envs)]
    finished_trajectories: List[Trajectory] = []

    (store_extra_information,
     current_players,
     legal_actions,
     num_agents,
     obs) = create_environment_variables(env, agent_vector)


    if show_progress:
        progress_bar = create_progress_bar(env, agent_vector, training, num_episodes, initial_episode)
    if summary_writer:
        # TODO: not sure these actually do what they claim
        logged_trajectories = 0
        action_time, handling_experience_time, env_step_time = 0., 0., 0.
        start_time = time()

    while len(finished_trajectories) < num_episodes:
        # Take action
        if summary_writer: action_time_start = time()
        action_vector = multienv_choose_action(
                agent_vector, env, obs, current_players, legal_actions)
        if summary_writer: action_time += time() - action_time_start

        # Environment step
        if summary_writer: env_step_time_start = time()
        succ_obs, rewards, dones, infos = env.step(action_vector)
        if summary_writer: env_step_time += time() - env_step_time_start

        # Update trajectories:
        if summary_writer:
            handling_experience_time_start = time()
        update_parallel_sequential_trajectories(ongoing_trajectories,
                            agent_vector, action_vector,
                            obs, rewards, succ_obs, dones,
                            current_players, store_extra_information)
        if summary_writer:
            handling_experience_time += time() - handling_experience_time_start

        # Update agents
        if training:
            propagate_experiences(agent_vector, ongoing_trajectories, store_extra_information)

        # Update observation
        obs = succ_obs

        # Update current players and legal actions
        legal_actions = [info.get('legal_actions', None) for info in infos]
        current_players = [info.get('current_player',
                                    (current_players[e_i] + 1) % num_agents)
                           for e_i, info in enumerate(infos)]

        # Handle with episode termination
        done_envs = [i for i in range(len(dones)) if dones[i]]
        if len(done_envs) > 0:
            # TODO: Figure out a way of nicely refactoring this
            if summary_writer: handling_experience_time_start = time()
            finished_trajectories, ongoing_trajectories, current_players = \
                    handle_finished_episodes(
                        training,
                        agent_vector,
                        ongoing_trajectories,
                        done_envs,
                        finished_trajectories,
                        current_players,
                        store_extra_information
                    )
            if summary_writer: handling_experience_time += time() - handling_experience_time_start
            if show_progress: progress_bar.update(len(done_envs))
            if summary_writer:
                logged_trajectories = log_end_of_episodes(
                    summary_writer,
                    finished_trajectories,
                    logged_trajectories,
                    initial_episode,
                    start_time,
                    action_time,
                    handling_experience_time,
                    env_step_time,
                )
                action_time, handling_experience_time, env_step_time = 0., 0., 0.

    if show_progress: progress_bar.close()
    return finished_trajectories


def log_end_of_episodes(summary_writer: SummaryWriter,
                        finished_trajectories: List[Trajectory],
                        logged_trajectories: int,
                        initial_episode: int,
                        start_time: float,
                        action_time: float,
                        handling_experience_time: float,
                        env_step_time: float):
    '''
    Writes to :param: summary_writer logs about :param: finished_trajectories

    :param logged_trajectories: More than 1 trajectory can be finished
                                concurrently, but we want one datapoint
                                to log for each one, so we have to keep
                                track of how many we've logged.
    '''
    finished_trajectories_lengths = list(map(lambda t: len(t), finished_trajectories))
    for i in range(logged_trajectories, len(finished_trajectories)):
        summary_writer.add_scalar('PerEpisode/Episode_length', finished_trajectories_lengths[i],
                                  initial_episode + (i+1))
        summary_writer.add_scalar('PerEpisode/Mean_episode_length', np.mean(finished_trajectories_lengths[:(i+1)]),
                                  initial_episode + (i+1))
        summary_writer.add_scalar('PerEpisode/Std_episode_length', np.std(finished_trajectories_lengths[:(i+1)]),
                                  initial_episode + (i+1))

    # Not sure if calculation is correct
    avg_time_per_episode = (time() - start_time) / len(finished_trajectories)

    summary_writer.add_scalar('Timing/Mean_time_per_episode', avg_time_per_episode,
                              initial_episode + (i+1))
    summary_writer.add_scalar('Timing/Take_action_time_taken', action_time,
                              initial_episode + (i+1))
    summary_writer.add_scalar('Timing/Handling_experience_time_taken', handling_experience_time,
                              initial_episode + (i+1))
    summary_writer.add_scalar('Timing/Env_step_time_taken', env_step_time,
                              initial_episode + (i+1))
    return len(finished_trajectories)


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
                             ongoing_trajectories: List[Trajectory],
                             done_envs: List[int],
                             finished_trajectories: List[Trajectory],
                             current_players: List[int],
                             store_extra_information: bool) \
                             -> Tuple[List[Trajectory], List[Trajectory]]:
    if training:
        propagate_last_experiences(agent_vector, ongoing_trajectories,
                                   done_envs, store_extra_information)
    # Reset players and trajectories
    # Why are we returning ongoing trajectories twice?
    ongoing_trajectories, finished_trajectories = update_finished_trajectories(
                    ongoing_trajectories, finished_trajectories, done_envs)
    current_players = reset_current_players(done_envs, current_players)
    return finished_trajectories, ongoing_trajectories, current_players


def reset_current_players(done_envs: List[int],
                          current_players: List[int]) -> List[int]:
    for i, e_i in enumerate(done_envs):
        current_players[e_i] = 0
    return current_players


def propagate_experiences(agent_vector: List[Agent],
                          trajectories: List[Trajectory],
                          store_extra_information: bool = False):
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
            agents_to_update,
            agent_to_update_per_env,
            trajectories,
            agent_vector,
            store_extra_information)

    propagate_batched_experiences(agent_experiences,
                                  agent_vector,
                                  environment_per_agents)


def propagate_batched_experiences(agent_experiences: Dict[int, List[Tuple]],
                                  agent_vector: List[Agent],
                                  environment_per_agents: Dict[int, List[int]]):
    '''
    Propagates :param: agent_experiences to the corresponding agents in
    :param: agent_vector, as dictated by :param: environment_per_agents
    '''
    for a_i, experiences in agent_experiences.items():
        if agent_vector[a_i].training:
            agent_vector[a_i].handle_multiple_experiences(
                    experiences, environment_per_agents[a_i])


def propagate_last_experiences(agent_vector: List[Agent],
                               trajectories: List[Trajectory],
                               done_envs: List[int],
                               store_extra_information: bool):
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
    # Potential refactoring by using `collect_agent_experiences_from_trajectories`
    for a_i, envs in environment_per_agents.items():
        for e_i in envs:
            (o, a, r, succ_o, d, extra_info) = extract_latest_experience_sequential_trajectory(
                    a_i, trajectories[e_i], store_extra_information)
            assert d, f'Episode should in environment {e_i} should be finished'
            agent_experiences[a_i] += [(o, a, r, succ_o, True, extra_info)]

    propagate_batched_experiences(agent_experiences,
                                  agent_vector, environment_per_agents)


def compute_agents_to_update_per_env(trajectories: List[Trajectory], done_envs, agent_vector):
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
                                                agent_vector: List[Agent],
                                                store_extra_information: bool) \
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
        experience = extract_latest_experience_sequential_trajectory(
            target_agent, trajectories[env_i], store_extra_information)
        agent_experiences[target_agent] += [experience]
    return agent_experiences


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


def create_progress_bar(env, agent_vector, training, num_episodes, initial_episode):
    agent_names = ', '.join([a.name for a in agent_vector])
    description = f'Simulating env {env.name} ({env.num_envs} processes). Agents [{agent_names}]. Training {training}'
    progress_bar = tqdm(total=num_episodes, desc=description, initial=initial_episode)
    return progress_bar


def create_environment_variables(env, agent_vector) -> Tuple:
    store_extra_information = any(
        [agent.requires_opponents_prediction or agent.requires_self_prediction
         for agent in agent_vector])

    current_players: List[int] = [0] * env.num_envs
    legal_actions: List[List] = [None] * env.num_envs # Revise
    num_agents = len(agent_vector)
    obs = env.reset()
    return store_extra_information, current_players, legal_actions, num_agents, obs
