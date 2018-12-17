import logging
import random
import numpy as np
from collections import namedtuple
BenchMarkStatistics = namedtuple('BenchMarkStatistics', 'iteration recorded_policy_vector winrates standard_deviations')

from concurrent.futures import as_completed

from multiagent_loops.simultaneous_action_rl_loop import run_episode


def benchmark_match_play_process(num_episodes, createNewEnvironment, benchmark_job, process_pool, matrix_queue, name):
    """
    :param num_episodes: Number of episodes used for stats collection
    :param createNewEnvironment OpenAI gym environment creation function
    :param benchmark_job: BenchmarkingJob containing iteration and policy vector to benchmark
    :param process_pool: ProcessPoolExecutor used to submit match runs jobs
    :param matrix_queue: Queue to which submit stats
    :param name: String identifying this benchmarking process
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.info('Started for {} episodes'.format(num_episodes))

    policy_vector = [recorded_policy.policy for recorded_policy in benchmark_job.recorded_policy_vector]

    # TODO Use given pool, but how?
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(single_match, *[createNewEnvironment(), policy_vector])
                   for _ in range(0, num_episodes)]

        wins_vector = [0, 0]
        for future in as_completed(futures):
            episode_winner = future.result()
            wins_vector[episode_winner] += 1
        winrates = [calculate_individual_winrate(wins_vector, i)
                    for i in range(len(policy_vector))]
        # TODO STD NOT WORKING
        standard_deviations = [calculate_individual_standarddeviation(wins_vector, winrates[i], i)
                               for i in range(len(policy_vector))]

    matrix_queue.put(BenchMarkStatistics(benchmark_job.iteration,
                                         benchmark_job.recorded_policy_vector,
                                         winrates, standard_deviations))
    logger.info('Benchmarking finished')


def calculate_individual_winrate(wins_vector, agent_index):
    return sum([wins_vector[agent_index] for i in range(len(wins_vector))]) / len(wins_vector)


def calculate_individual_standarddeviation(wins_vector, mean_agent_winrate, agent_index):
    variance = sum(map(lambda x: (x - mean_agent_winrate)**2, [wins_vector[agent_index] for i in range(len(wins_vector))])) / (len(wins_vector) - 1)
    return np.sqrt(variance)


def single_match(env, policy_vector):
    # trajectory: [(s,a,r,s')]
    trajectory = run_episode(env, policy_vector, training=False)
    reward_vector = lambda t: t[2]
    individal_policy_trajectory_reward = lambda t, agent_index: sum(map(lambda experience: reward_vector(experience)[agent_index], t))
    cumulative_reward_vector = [individal_policy_trajectory_reward(trajectory, i) for i in range(len(policy_vector))]
    episode_winner = choose_winner(cumulative_reward_vector)
    return episode_winner


def choose_winner(cumulative_reward_vector, break_ties=random.choice):
    indexes_max_score = np.argwhere(cumulative_reward_vector == np.amax(cumulative_reward_vector))
    return break_ties(indexes_max_score.flatten().tolist())

    

