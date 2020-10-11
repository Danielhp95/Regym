from typing import List, Tuple, Any, Dict
from copy import deepcopy

import numpy as np
import gym

import regym
from regym.rl_loops import Trajectory
from regym.rl_algorithms.agents import Agent


def run_episode(env: gym.Env, agent_vector: List[Agent], training: bool,
                render_mode: str) -> Trajectory:
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
    store_extra_information = any([agent.requires_opponents_prediction for agent in agent_vector])

    trajectory = Trajectory(env_type=regym.environments.EnvType.MULTIAGENT_SEQUENTIAL_ACTION,
                            num_agents=len(agent_vector))
    observations, done = env.reset(), False
    current_player = 0  # Assumption: The first agent to act is always the 0th agent
    # Unfortunately, OpenAIGym does not have a standardized interface
    # To support which actions are legal at an initial state. These can only be extracted
    # via the "info" dictionary given by the env.step(...) function
    # Thus: Assumption: all actions are permitted on the first state
    legal_actions: List = None
    while not done:
        if render_mode: env.render(mode=render_mode)
        agent = agent_vector[current_player]

        # Take action
        action = choose_action(agent, env, observations[current_player], current_player, legal_actions)

        # Environment step
        succ_observations, reward_vector, done, info = env.step(action)

        extra_info = extract_extra_info(store_extra_information, current_player, agent_vector)

        trajectory.add_timestep(observations, action, reward_vector, succ_observations, done,
                                acting_agents=[current_player],
                                extra_info=extra_info)

        # Update agents
        if training: propagate_experience(agent_vector, trajectory,
                                          reward_vector, succ_observations, done)

        # Update observation
        observations = succ_observations

        if 'legal_actions' in info: legal_actions = info['legal_actions']
        if 'current_player' in info: current_player = info['current_player']
        else: current_player = (current_player + 1) % len(agent_vector)

    if training: propagate_last_experience(agent_vector, trajectory)
    return trajectory

def extract_extra_info(store_extra_information, current_player, agent_vector):
    extra_info = None
    if store_extra_information:
        extra_info = {current_player: agent_vector[current_player].current_prediction}
    else: extra_info = None
    return extra_info


def choose_action(agent, env, observation, current_player, legal_actions):
    '''
    TODO: document that we are using this for both async and normal?
    '''
    if not agent.requires_environment_model:
        action = agent.model_free_take_action(observation,
                                              legal_actions=legal_actions,
                                              multi_action=False)
    else:
        action = agent.model_based_take_action(deepcopy(env), observation,
                                               current_player,
                                               multi_action=False)
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
        (o, a, r, succ_o, d) = extract_latest_experience(agent_to_update, trajectory, agent_vector)
        agent_vector[agent_to_update].handle_experience(o, a, r, succ_o, d)


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
    o, a = get_last_observation_and_action_for_agent(agent_id,
                                                     trajectory,
                                                     len(agent_vector))
    (_, _, reward, succ_observation, done) = trajectory[-1]
    return (o, a, reward[agent_id], succ_observation[agent_id], done)


def propagate_last_experience(agent_vector: List, trajectory: List):
    '''
    Sequential environments will often feature a terminal state which yields
    a reward signal to each agent (i.e how much each agent wins / loses on poker).

    This function propagates this reward signal to all agents
    who have not received it.

    :param agent_vector: List of agents acting in current environment
    :param trajectory: Current (finished) episode trajectory
    '''
    reward_vector = trajectory[-1][2]
    succ_observations = trajectory[-1][3]
    agent_indices = list(range(len(agent_vector)))
    # This agent already processed the last experience
    agent_indices.pop(len(trajectory) % len(agent_vector))

    for i in agent_indices:
        if not agent_vector[i].training: continue
        o, a = get_last_observation_and_action_for_agent(i, trajectory,
                                                         len(agent_vector))
        agent_vector[i].handle_experience(o, a, reward_vector[i],
                                          succ_observations[i], done=True)


def get_last_observation_and_action_for_agent(target_agent_id: int,
                                              trajectory: List, num_agents: int) -> Tuple:
    '''
    # TODO: assume games where turns are taken in cyclic fashion.

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
