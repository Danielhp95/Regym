from typing import List, Tuple, Any, Dict
from copy import deepcopy

import numpy as np
import gym

import regym
from regym.rl_algorithms.agents import Agent
from regym.environments.tasks import RegymAsyncVectorEnv

from regym.rl_loops.utils import update_trajectories, update_parallel_sequential_trajectories, update_finished_trajectories


def run_episode(env: gym.Env, agent_vector: List, training: bool, render_mode: str):
    '''
    Runs a single multi-agent rl loop until termination for a sequential environment

    ASSUMES that turns are taken in clockwise fashion:
        - Player 1 acts, player 2 acts..., player n acts, player 1 acts...
        - where n is the length of :param: agent_vector

    :param env: OpenAI gym environment
    :param agent_vector: Vector containing the agent for each agent in the environment
    :param training: (boolean) Whether the agents will learn from the experience they recieve

    :param render_mode: TODO: add explanation
    :returns: Episode trajectory (o,a,r,o')
    '''
    observations, done, trajectory = env.reset(), False, []
    current_player = 0  # Assumption: The first agent to act is always the 0th agent
    # Unfortunately, OpenAIGym does not have a standardized interface
    # To support which actions are legal at an initial state. These can only be extracted
    # via the "info" dictionary given by the env.step(...) function
    # Thus: Assumption: all actions are permitted on the first state
    legal_actions: List
    while not done:
        agent = agent_vector[current_player]

        # Take action
        action = choose_action(agent, env, observations[current_player], current_player, legal_actions)

        # Environment step
        succ_observations, reward_vector, done, info = env.step(action)
        trajectory.append((observations, action, reward_vector, succ_observations, done))

        # Update agents
        if training: propagate_experience(agent_vector, trajectory,
                                          reward_vector, succ_observations, done)

        # Update observation
        observations = succ_observations

        # If environment provides information about next player, use it
        # otherwise, assume that players' turn rotate circularly. 
        if 'current_player' in info: current_player = info['current_player']
        else: current_player = (current_player + 1) % len(agent_vector)

        if 'legal_actions' in info: legal_actions = info['legal_actions']

    if training: propagate_last_experience(agent_vector, trajectory, reward_vector, succ_observations)
    return trajectory


def async_run_episode(env: RegymAsyncVectorEnv, agent_vector: List, training: bool,
                      num_episodes: int) \
                      -> List[List[Tuple[Any, Any, Any, Any, bool]]]:
    '''
    TODO: document, refactor
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
        action_vector = multi_env_choose_action(
                agent_vector, env, obs, current_players, legal_actions)

        # Environment step
        succ_obs, rewards, dones, infos = env.step(action_vector)

        # Update trajectories:
        update_parallel_sequential_trajectories(ongoing_trajectories, current_players,
                action_vector, obs, rewards, succ_obs, dones)

        # Update agents
        #if training:
        #    for i in range(len(ongoing_trajectories)):
        #        propagate_experience(agent_vector, ongoing_trajectories[i],
        #                             rewards[i], succ_obs[i], dones[i])

        done_envs = update_finished_trajectories(ongoing_trajectories,
                                                 finished_trajectories, dones)

        # Update current players and legal actions
        legal_actions = [info.get('legal_actions', None) for info in infos]
        current_players = [info.get('current_player',
                                    (current_players[e_i] + 1) % num_agents)
                           for e_i, info in enumerate(infos)]
        for e_i in done_envs: current_players[e_i] = 0  # This might break?

    return finished_trajectories


def multi_env_choose_action(agent_vector, env: RegymAsyncVectorEnv, obs,
                            current_players, legal_actions):
    action_vector = [None] * env.num_envs
    # Find indices of which envs each player should play, on a dict
    agent_signals = extract_signals_for_acting_agents(
            agent_vector, env, obs, current_players, legal_actions)

    for a_i, signals in agent_signals.items():
        a = agent_vector[a_i]
        if not a.requires_environment_model:
            partial_action_vector = a.model_free_take_action(
                    signals['obs'], legal_actions=signals['legal_actions'])
        else:
            raise NotImplementedError('Gimme a minute')
        # fill action_vector
        for env_id, action in zip(signals['env_ids'], partial_action_vector):
            assert action_vector[env_id] is None
            action_vector[env_id] = action
    return action_vector


def extract_signals_for_acting_agents(agent_vector, env, obs,
                                      current_players, legal_actions) \
                                              -> Dict[int, Dict[str, List]]:
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


def choose_action(agent, env, observation, current_player, legal_actions):
    '''
    TODO: document that we are using this for both async and normal?
    '''
    if not agent.requires_environment_model:
        action = agent.model_free_take_action(observation,
                                              legal_actions=legal_actions)
    else:
        action = agent.model_based_take_action(deepcopy(env), observation, current_player)
    return action


def propagate_experience(agent_vector: List, trajectory: List,
                         reward_vector: List, succ_observations: List,
                         done: bool):
    '''
    Propagates the experience tuple to the corresponding agent

    ASSUMPTION: Agents take action in a circular fashion:
        Action order: 0 -> 1 -> ... -> number_agents -> 0 -> 1 -> ...
    :param agent_vector: List of agents acting in current environment
    :param reward_vector: Reward vector for environment step
    :param succ_observations: Succesor observations for current env step
    :param done: Termination flag, whether the episode has finished
    '''
    if len(trajectory) < len(agent_vector): return

    agent_to_update = len(trajectory) % len(agent_vector)
    if agent_vector[agent_to_update].training:
        update_agent(agent_to_update, trajectory, agent_vector,
                     reward_vector[agent_to_update],
                     succ_observations[agent_to_update], done)


def update_agent(agent_id: int, trajectory: List, agent_vector: List,
                 reward: float, succ_observation: np.ndarray, done: bool):
    '''
    ASSUMPTION: every non-terminal observation corresponds to
    the an information set unique for the player whose turn it is.
    This means that each "experience" is from which an RL agent will learn
    (o, a, r, o') is fragmented throughout the trajectory. This function
    "stiches together" the right environmental signals, ensuring that
    each agent only has access to information from their own information sets.

    :param agent_id: Index of agent which will receive a new experience
    :param trajectory: Current episode trajectory
    :param agent_vector: List of agents acting in current environment
    :param reward: Scalar reward for :param: agent_id
    :param succ_observation: Succesor observation for :param: agent_id
    :param done: Termination flag, whether the episode has finished
    '''
    o, a = get_last_observation_and_action_for_agent(agent_id,
                                                     trajectory,
                                                     len(agent_vector))
    experience = (o, a, reward, succ_observation, done)
    agent_vector[agent_id].handle_experience(*experience)


def propagate_last_experience(agent_vector: List, trajectory: List,
                              reward_vector: List[float],
                              succ_observations: List[np.ndarray]):
    '''
    Sequential environments will often feature a terminal state which yields
    a reward signal to each agent (i.e how much each agent wins / loses on poker).

    This function propagates this reward signal to all agents
    who have not received it.

    :param agent_vector: List of agents acting in current environment
    :param trajectory: Current (finished) episode trajectory
    :param reward_vector: Reward vector for terminal environment step
    :param succ_observations: Succesor observations for current env step
    '''
    agent_indices = list(range(len(agent_vector)))
    # This agent already processed the last experience
    agent_indices.pop(len(trajectory) % len(agent_vector))

    for i in agent_indices:
        if not agent_vector[i].training: continue
        update_agent(i, trajectory, agent_vector, reward_vector[i],
                     succ_observations[i], True)


def get_last_observation_and_action_for_agent(target_agent_id: int,
                                              trajectory: List, num_agents: int) -> Tuple:
    '''
    Obtains the last observation and action for agent :param: target_agent_id
    from the :param: trajectory.

    :param target_agent_id: Index of agent whose last observation / action
                            we are searching for
    :param trajectory: Sequence of (o_i,a_i,r_i,o'_{i+1}) for all players i.
    :param num_agents: Number of agents acting in the current environment
    :returns: The last observation (information state) and action taken
              at such observation by player :param: target_agent_id.
    '''
    last_agent_to_act = (len(trajectory) - 1) % num_agents
    # Offsets are negative, exploiting Python's backwards index lookup
    previous_timestep = trajectory[-num_agents]
    last_observation = previous_timestep[0][target_agent_id]
    last_action = previous_timestep[1]
    return last_observation, last_action
