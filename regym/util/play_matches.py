from typing import List
import random
import numpy as np

from regym.environments import Task


def play_multiple_matches(task: Task, agent_vector: List, n_matches: int, keep_trajectories=False):
    '''
    Computes a winrate vector by making :param agent_vector: play in :param env:
    for :param n_matches:. If :param keep_trajectories: is True, a tuple is returned
    where the first element is the winrate vector and the second is the vector of
    trajectories.

    :param task: regym Task containing an OpenAI Gym environment where the matches wll be run
    :param agent_vector: vector of agents capable of acting in :param env:
    :param n_matches: number of matches to be played
    :returns: Vector containing the winrate for each agent
    '''
    winrates = np.zeros(task.num_agents)
    trajectories = []
    for episode in range(n_matches):
        if keep_trajectories:
            winner, trajectory = play_single_match(task, agent_vector, True)
            trajectories.append(trajectory)
        else:
            winner = play_single_match(task, agent_vector, False)
        winrates[winner] += 1
    winrates /= n_matches
    if len(trajectories) == 0: return winrates
    else: return winrates, trajectories


def play_single_match(task, agent_vector, keep_trajectories=False):
    trajectory = task.run_episode(agent_vector, training=False)
    episode_winner = extract_winner(trajectory)
    if keep_trajectories: return episode_winner, trajectory
    else: return episode_winner


def extract_winner(trajectory, break_ties=random.choice):
    cumulative_reward_vector = extract_cumulative_rewards(trajectory)
    indexes_max_score = np.argwhere(cumulative_reward_vector == np.amax(cumulative_reward_vector))
    return break_ties(indexes_max_score.flatten().tolist())


def extract_cumulative_rewards(trajectory):
    reward_vector = lambda t: t[2]
    number_of_agents = len(reward_vector(trajectory[0])) if isinstance(reward_vector(trajectory[0]), list) else 1
    if number_of_agents == 1:
        cum_reward = sum(map(lambda experience: reward_vector(experience),
                             trajectory))
    else:
        individal_agent_trajectory_reward = lambda t, agent_index: sum(map(lambda experience: reward_vector(experience)[agent_index], t))
        cum_reward = [individal_agent_trajectory_reward(trajectory, i) for i in range(number_of_agents)]
    return cum_reward
