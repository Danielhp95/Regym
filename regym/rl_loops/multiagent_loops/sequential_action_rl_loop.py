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

        extra_info = collect_extra_information_from_agent(store_extra_information, current_player, agent_vector)

        trajectory.add_timestep(observations, action, reward_vector, succ_observations, done,
                                acting_agents=[current_player],
                                extra_info=extra_info)

        # Update agents
        if training: propagate_experience(agent_vector, trajectory)

        # Update observation
        observations = succ_observations

        if 'legal_actions' in info: legal_actions = info['legal_actions']
        if 'current_player' in info: current_player = info['current_player']
        else: current_player = (current_player + 1) % len(agent_vector)

    if training: propagate_last_experience(agent_vector, trajectory)
    return trajectory


def collect_extra_information_from_agent(store_extra_information: bool, current_player: int,
                       agent_vector: List[Agent]) -> Dict[int, Dict[str, Any]]:
    if store_extra_information:
        extra_info = {current_player: agent_vector[current_player].current_prediction}
    else: extra_info = {}
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


def propagate_experience(agent_vector: List, trajectory: Trajectory):
    '''
    Propagates the experience tuple to the corresponding agent

    ASSUMPTION: Agents take action in a circular fashion:
        Action order: 0 -> 1 -> ... -> number_agents -> 0 -> 1 -> ...
    :param agent_vector: List of agents acting in current environment
    :param trajectory: Current episode trajectory
    '''
    if len(trajectory) < len(agent_vector): return

    agent_to_update = len(trajectory) % len(agent_vector)  # Cyclic assumption
    if agent_vector[agent_to_update].training:
        (o, a, r, succ_o, d, extra_info) = extract_latest_experience(agent_to_update, trajectory)
        agent_vector[agent_to_update].handle_experience(o, a, r, succ_o, d, extra_info)


def propagate_last_experience(agent_vector: List, trajectory: Trajectory):
    '''
    Sequential environments will often feature a terminal state which yields
    a reward signal to each agent (i.e how much each agent wins / loses on poker).

    This function propagates this reward signal to all agents
    who have not received it, because they have not taken an acion on the
    environment state leading up to it.

    :param agent_vector: List of agents acting in current environment
    :param trajectory: Current (finished) episode trajectory
    '''
    agent_indices = list(range(len(agent_vector)))
    # This agent already processed the last experience
    agent_indices.pop(len(trajectory) % len(agent_vector))

    for a_i in agent_indices:
        if not agent_vector[a_i].training: continue
        (o, a, r, succ_o, d, extra_info) = extract_latest_experience(a_i, trajectory)
        assert d, 'Episode should be finished'
        agent_vector[a_i].handle_experience(o, a, r, succ_o, d, extra_info)


def extract_latest_experience(agent_id: int, trajectory: Trajectory) -> Tuple:
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
    '''
    t1 = trajectory.last_acting_timestep_for_agent(agent_id)
    t2 = trajectory[-1]

    o, succ_o = t1.observation[agent_id], t2.succ_observation[agent_id]
    a = t1.action
    r = t1.reward[agent_id]  # Perhaps there's a better way of computting agent's reward?
                             # Such as summing up all intermediate rewards for :agent_id:
                             # between t1.t and t2.t?
    done = t2.done

    extra_info = extract_extra_info_from_trajectory(agent_id, trajectory)
    return (o, a, r, succ_o, done, extra_info)


def extract_extra_info_from_trajectory(agent_id: int,
                                       trajectory: Trajectory) \
                            -> Dict[int, Dict[str, Any]]:
    '''
    TODO: mention about stiching together current predicions from trajectory
    '''
    time_index_after_agent_action = -(trajectory.num_agents) + 1
    relevant_timesteps = trajectory[time_index_after_agent_action:]

    extra_info = {}
    for timestep in relevant_timesteps:
        for a_i, v in timestep.extra_info.items():
            if a_i == agent_id:
                # An agent should not be fed info about it's own predictions
                continue
            assert a_i not in extra_info, 'Breaking cyclic turn assumption'
            extra_info[a_i] = v
    return extra_info
