from functools import reduce
from tqdm import tqdm
import numpy as np
from regym.environments import generate_task
from regym.rl_loops.multiagent_loops import sequential_action_rl_loop
from regym.rl_loops.multiagent_loops import simultaneous_action_rl_loop


def learn_against_fix_opponent(agent, fixed_opponent,
                               env: str, env_type: str,
                               agent_position: int,
                               total_episodes: int, training_percentage: float,
                               reward_tolerance: float,
                               maximum_average_reward: float,
                               evaluation_method: str):
    '''
    Test used to :assert: that :param: agent is 'learning' by
    learning a best response against a fixed agent.

    :param agent: Agent which will train against a fixed opponent
    :param fixed_opponent: Agent with a fixed (frozen) policy which
                           will play against :param: agent
    :param env: String ID of the environment where the agents will paly
    :param env_type: Either "sequential" or "simultaneous"
    :param total_episodes: Number of episodes used for training + evaluation
    :param training_percentage: % of :param total_episodes: that will be used
                                to train :param: agent.
    :param reward_tolerance: Tolerance (epsilon) allowed when considering if
                             :param: agent has solved the environment.
    :param maximum_average_reward: Maximum average reward per episode
    :param evaluation_method: Whether to consider 'average' trajectory
                              or only the 'last' reward.
    '''
    env = generate_task(env).env
    maximum_average_reward = maximum_average_reward
    if env_type == 'simultaneous': run_episode_func = simultaneous_action_rl_loop.run_episode
    elif env_type == 'sequential': run_episode_func = sequential_action_rl_loop.run_episode
    else: raise ValueError(f"Parameter env_type must be 'sequential' or 'simultaneous'. Found {env_type}")

    training_episodes = int(total_episodes * training_percentage)
    inference_episodes = total_episodes - training_episodes

    training_trajectories = simulate(env, agent, fixed_opponent, agent_position,
                                     episodes=training_episodes, training=True,
                                     run_episode_func=run_episode_func)
    agent.training = False
    inference_trajectories = simulate(env, agent, fixed_opponent, agent_position,
                                     episodes=inference_episodes, training=False,
                                     run_episode_func=run_episode_func)

    if evaluation_method == 'average':
        inference_reward = average_reward(inference_trajectories,
                                          agent_position)
    elif evaluation_method == 'last':
        inference_reward = sum(map(lambda t: last_trajectory_reward(t, agent_position),
                                   inference_trajectories))
        inference_reward /= float(len(inference_trajectories))

    reward_threshold = maximum_average_reward - reward_tolerance
    assert inference_reward >= reward_threshold, \
           f'Reward obtained during inference wasn\'t high enough\n{inference_reward} < {reward_threshold}'


def simulate(env, agent, fixed_opponent, agent_position, episodes, training, run_episode_func):
    agent_vector = [fixed_opponent]
    agent_vector.insert(agent_position, agent)
    agent_names = f'{agent_vector[0].name} vs {agent_vector[1].name}'
    trajectories = list()
    mode = 'TRAINING' if training else 'INFERENCE'
    progress_bar = tqdm(range(episodes))
    for e in progress_bar:
        trajectory = run_episode_func(env, agent_vector, training=training)
        trajectories.append(trajectory)
        traj_reward = trajectory_reward(trajectory, agent_position)
        progress_bar.set_description(f'{mode} [{env.spec.id}] [{agent_names}]. Last trajectory reward: {traj_reward}')
    return trajectories


def average_reward(trajectories, agent_position):
    rewards = sum(map(lambda t: trajectory_reward(t, agent_position),
                     trajectories))
    return rewards / float(len(trajectories))


def trajectory_reward(trajectory, agent_position):
    return sum(map(lambda experience: experience[2][agent_position], trajectory))


def last_trajectory_reward(trajectory, agent_position):
    return trajectory[-1][2][agent_position]
