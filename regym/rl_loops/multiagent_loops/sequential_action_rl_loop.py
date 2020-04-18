from typing import List, Tuple
from copy import deepcopy
import numpy as np
import gym


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
    legal_actions: List = None
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


def choose_action(agent, env, observation, current_player, legal_actions):
    if not agent.requires_environment_model:
        action = agent.model_free_take_action(observation,
                                              legal_actions=legal_actions)
    else:
        action = agent.model_based_take_action(deepcopy(env), observation, current_player)
    return action


def propagate_experience(agent_vector: List, trajectory: List, reward_vector: List, succ_observations: List, done: bool):
    if len(trajectory) >= len(agent_vector):
        agent_to_update = len(trajectory) % len(agent_vector)
        if agent_vector[agent_to_update].training:
            update_agent(agent_to_update, trajectory, agent_vector,
                         reward_vector[agent_to_update],
                         succ_observations[agent_to_update], done)


def update_agent(agent_id: int, trajectory: List, agent_vector: List,
                 reward: float, succ_observation: np.ndarray, done: bool):
    '''
    This function assumes that every non-terminal observation corresponds to
    the an information set uniquely for the player whose turn it is.
    This means that each "experience" is from which an RL agent will learn
    (o, a, r, o') is fragmented throughout the trajectory. This function
    "stiches together" the right environmental signals, ensuring that
    each agent only has access to information from their own information sets.
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
    :param target_agent_id: Index of agent whose last observation / action
                            we are searching for
    :param trajectory: Sequence of (o_i,a_i,r_i,o'_{i+1}) for all players i.
    :returns: The last observation (information state) and action taken
              at such observation by player :param: target_agent_id.
    '''
    last_agent_to_act = (len(trajectory) - 1) % num_agents
    if last_agent_to_act >= target_agent_id:
        offset = target_agent_id - last_agent_to_act - 1
    else:
        offset = num_agents - (target_agent_id - last_agent_to_act) - 1
    previous_timestep = trajectory[offset]
    last_observation = previous_timestep[0][target_agent_id]
    last_action = previous_timestep[1]
    return last_observation, last_action
