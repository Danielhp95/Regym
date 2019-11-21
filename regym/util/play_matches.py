import random
import numpy as np


def play_multiple_matches(env, agent_vector, n_matches: int):
    '''
    :param agent_vector: vector of agents capable of acting in :param env:
    :returns: Vector containing the winrate for each agent
    '''
    winrates = np.zeros(len(agent_vector))
    for episode in range(n_matches):
        winner = play_single_match(env, agent_vector)
        winrates[winner] += 1
    winrates /= n_matches
    return winrates


def play_single_match(env, agent_vector):
    # TODO: find better way of calculating who won.
    # trajectory: [(s,a,r,s')]
    from regym.rl_loops.multiagent_loops.simultaneous_action_rl_loop import run_episode
    trajectory = run_episode(env, agent_vector, training=False)
    episode_winner = extract_winner(trajectory)
    return episode_winner


def extract_winner(trajectory, break_ties=random.choice):
    reward_vector = lambda t: t[2]
    number_of_agents = len(reward_vector(trajectory[0]))
    individal_agent_trajectory_reward = lambda t, agent_index: sum(map(lambda experience: reward_vector(experience)[agent_index], t))
    cumulative_reward_vector = [individal_agent_trajectory_reward(trajectory, i) for i in range(number_of_agents)]
    indexes_max_score = np.argwhere(cumulative_reward_vector == np.amax(cumulative_reward_vector))
    return break_ties(indexes_max_score.flatten().tolist())
