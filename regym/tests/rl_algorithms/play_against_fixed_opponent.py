from typing import Optional, List
from functools import reduce
import math

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from regym.environments import Task
from regym.util import extract_winner
from regym.rl_algorithms.agents import Agent

# TODO: consider moving some of these functions to regym.utils


def learn_against_fix_opponent(agent: Agent, fixed_opponent: Agent,
                               task: Task,
                               agent_position: int,
                               training_episodes: int,
                               test_episodes: int,
                               reward_tolerance: float,
                               maximum_average_reward: float,
                               evaluation_method: str,
                               alter_agent_positions: bool = False,
                               benchmarking_episodes: int = 0,
                               benchmark_every_n_episodes: int = 0,
                               summary_writer: Optional[SummaryWriter] = None):
    '''
    Test used to :assert: that :param: agent is 'learning' by
    learning a best response against a fixed agent.

    :param agent: Agent which will train against a fixed opponent
    :param fixed_opponent: Agent with a fixed (frozen) policy which
                           will play against :param: agent
    :param task: Task where the agents will play
    :param env_type: Either "sequential" or "simultaneous"
    :param training_episodes: Number of episodes used for training
    :param test_episodes: Number of episodes used for testing
    :param benchmark_every_n_episodes: Number of episodes that the agent will
                                       train before a new benchmarking commences
    :param benchmarking_episodes: Number of episodes used to benchmark
                                  the agent on every benchmarking checkpoint.
    :param reward_tolerance: Tolerance (epsilon) allowed when considering if
                             :param: agent has solved the environment.
    :param maximum_average_reward: Maximum average reward per episode
    :param evaluation_method: Whether to consider 'average' trajectory
                              or only the 'last' reward.
    :param alter_agent_positions: Whether to randomly select agent positions
                                  after each episode
    :param summary_writer: Torch SummaryWriter to log info during
                           training and benchmarking.
    '''
    training_trajectories = train_and_benchmark(task, agent, fixed_opponent,
                                                training_episodes,
                                                benchmark_every_n_episodes,
                                                benchmarking_episodes,
                                                agent_position,
                                                alter_agent_positions,
                                                summary_writer)

    agent.training = False
    test_trajectories = simulate(task, agent, fixed_opponent, agent_position,
                                 episodes=test_episodes, training=False,
                                 mode='TESTING')

    test_reward = extract_test_reward(evaluation_method, test_trajectories, agent_position)

    reward_threshold = maximum_average_reward - reward_tolerance
    assert test_reward >= reward_threshold, \
           f'Reward obtained during inference wasn\'t high enough\n{test_reward} < {reward_threshold}'


def simulate(task: Task, agent: Agent, fixed_opponent: Agent,
             agent_position: int, episodes: int,
             training, mode: str) -> List:
    agent_vector = [fixed_opponent]
    agent_vector.insert(agent_position, agent)
    agent_names = f'{agent_vector[0].name} vs {agent_vector[1].name}'
    trajectories = list()
    progress_bar = tqdm(range(episodes))
    for e in progress_bar:
        trajectory = task.run_episode(agent_vector, training=training)
        trajectories.append(trajectory)
        traj_reward = trajectory_reward(trajectory, agent_position)
        progress_bar.set_description(f'{mode} [{task.name}] [{agent_names}]. Last trajectory reward: {traj_reward}')
    return trajectories


def train_and_benchmark(task, agent: Agent, fixed_opponent: Agent,
                        training_episodes: int, benchmark_every_n_episodes: int,
                        benchmarking_episodes: int,
                        agent_position: int, alter_agent_positions: bool,
                        summary_writer: Optional[SummaryWriter]):
    training_trajectories = list()
    # This might be bugged
    for e in range(0, training_episodes, benchmark_every_n_episodes):
        training_trajectories += simulate(task, agent, fixed_opponent,
                                          agent_position,
                                          episodes=benchmark_every_n_episodes,
                                          training=True, mode='TRAINING')
        agent.training = False
        benchmark_agent(task, agent, fixed_opponent, agent_position,
                        starting_episode=(e + benchmark_every_n_episodes),
                        episodes=benchmarking_episodes,
                        summary_writer=summary_writer)
        torch.save(agent, f'{agent.name}_{task.name}_{e + benchmark_every_n_episodes}.pt')
        agent.training = True
    return training_trajectories


def benchmark_agent(task: Task, agent: Agent, fixed_opponent: Agent,
                    agent_position, episodes,
                    starting_episode: int,
                    summary_writer: Optional[SummaryWriter]):
    trajectories = simulate(task, agent, fixed_opponent, agent_position,
                            episodes, training=False, mode='BENCHMARKING')

    # How can we also print this info in a useful way?
    winrate = len(list(filter(lambda t: extract_winner(t) == agent_position,
                              trajectories))) / len(trajectories)
    avg_episode_length = reduce(lambda acc, t: acc + len(t), trajectories, 0) / len(trajectories)
    avg_episode_reward = reduce(lambda acc, t: acc + trajectory_reward(t, agent_position),
                                trajectories, 0) / len(trajectories)
    if summary_writer:
        summary_writer.add_scalar('Benchmarking/Winrate', winrate, starting_episode)
        summary_writer.add_scalar('Benchmarking/Average_episode_length', avg_episode_length, starting_episode)
        summary_writer.add_scalar('Benchmarking/Average_episode_reward', avg_episode_reward, starting_episode)


def extract_test_reward(evaluation_method, test_trajectories, agent_position):
    if evaluation_method == 'cumulative':
        test_reward = average_reward(test_trajectories,
                                          agent_position)

    elif evaluation_method == 'last':
        test_reward = sum(map(lambda t: last_trajectory_reward(t, agent_position),
                                   test_trajectories))
        test_reward /= float(len(test_trajectories))
    return test_reward


def average_reward(trajectories, agent_position):
    rewards = sum(map(lambda t: trajectory_reward(t, agent_position),
                     trajectories))
    return rewards / float(len(trajectories))


def trajectory_reward(trajectory, agent_position):
    return sum(map(lambda experience: experience[2][agent_position], trajectory))


def last_trajectory_reward(trajectory, agent_position):
    return trajectory[-1][2][agent_position]
