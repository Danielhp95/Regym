from typing import List
import numpy as np
import gym


def run_episode(env: gym.Env, agent_vector: List, training: bool):
    '''
    Runs a single multi-agent rl loop until termination for a sequential environment

    ASSUMES that turns are taken in clockwise fashion:
        - Player 1 acts, player 2 acts..., player n acts, player 1 acts...
        - where n is the length of :param: agent_vector

    :param env: OpenAI gym environment
    :param agent_vector: Vector containing the agent for each agent in the environment
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :returns: Episode trajectory (o,a,r,o')
    '''
    observations, done, trajectory, current_player = env.reset(), False, [], 0
    while not done:
        agent = agent_vector[current_player]
        action = agent.take_action(observations[current_player])
        succ_observations, reward_vector, done, _ = env.step(action)
        trajectory.append((observations, action, reward_vector, succ_observations, done))

        if training and len(trajectory) >= len(agent_vector):
            agent_to_update = len(trajectory) % len(agent_vector)
            update_agent(agent_to_update, trajectory, agent_vector,
                         reward_vector[agent_to_update],
                         succ_observations[agent_to_update], done)

        observations = succ_observations
        current_player = (current_player + 1) % len(agent_vector)

    propagate_last_experience(agent_vector, trajectory, reward_vector, succ_observations)
    return trajectory


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
    agents_to_update = list(range(len(agent_vector)))
    # This agent already processed the last experience
    agents_to_update.pop(len(trajectory) % len(agent_vector))

    for i in agents_to_update:
        update_agent(i, trajectory, agent_vector, reward_vector[i],
                     succ_observations[i], True)


def get_last_observation_and_action_for_agent(target_agent_id: int,
                                              trajectory: List, num_agents: int):
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
