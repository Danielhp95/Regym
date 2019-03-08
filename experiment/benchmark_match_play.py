import time
import logging
import logging.handlers
import random
import numpy as np
from collections import namedtuple

from torch.multiprocessing import Process
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor

from rl_algorithms import AgentHook
from multiagent_loops.simultaneous_action_rl_loop import run_episode

BenchMarkStatistics = namedtuple('BenchMarkStatistics', 'iteration recorded_agent_vector winrates')


def benchmark_match_play_process(num_episodes, createNewEnvironment, benchmark_job, process_pool, matrix_queue, name):
    """
    :param num_episodes: Number of episodes used for stats collection
    :param createNewEnvironment OpenAI gym environment creation function
    :param benchmark_job: BenchmarkingJob containing iteration and agent vector to benchmark
    :param process_pool: ProcessPoolExecutor used to submit match runs jobs
    :param matrix_queue: Queue to which submit stats
    :param name: String identifying this benchmarking process
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.handlers.SocketHandler(host='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT))
    logger.info('Started for {} episodes'.format(num_episodes))

    agent_vector = [recorded_agent.agent for recorded_agent in benchmark_job.recorded_agent_vector]

    # TODO Use given pool, but how?
    with ProcessPoolExecutor(max_workers=3) as executor:
        benchmark_start = time.time()
        futures = [executor.submit(single_match, *[createNewEnvironment, agent_vector])
                   for _ in range(num_episodes)]

        wins_vector = [0 for _ in range(len(agent_vector))]

        for future in as_completed(futures):
            episode_winner = future.result()
            wins_vector[episode_winner] += 1
        benchmark_duration = time.time() - benchmark_start
        winrates = [winrate / num_episodes for winrate in wins_vector]

    matrix_queue.put(BenchMarkStatistics(benchmark_job.iteration,
                                         benchmark_job.recorded_agent_vector,
                                         winrates))
    logger.info('Benchmarking finished. Duration: {} seconds'.format(benchmark_duration))
    matrix_queue.join()


def single_match(createNewEnvironment, agent_vector):
    # trajectory: [(s,a,r,s')]
    unhooked_agents = [AgentHook.unhook(agent) for agent in agent_vector]
    trajectory = run_episode(createNewEnvironment(), unhooked_agents, training=False)
    reward_vector = lambda t: t[2]
    individal_agent_trajectory_reward = lambda t, agent_index: sum(map(lambda experience: reward_vector(experience)[agent_index], t))
    cumulative_reward_vector = [individal_agent_trajectory_reward(trajectory, i) for i in range(len(agent_vector))]
    episode_winner = choose_winner(cumulative_reward_vector)
    return episode_winner


def choose_winner(cumulative_reward_vector, break_ties=random.choice):
    indexes_max_score = np.argwhere(cumulative_reward_vector == np.amax(cumulative_reward_vector))
    return break_ties(indexes_max_score.flatten().tolist())


def create_benchmark_process(benchmarking_episodes, createNewEnvironment, benchmark_job, pool, matrix_queue, name):
    """
    Creates a benchmarking process for the precomputed benchmark_job.
    The results of the benchmark will be put in the matrix_queue to populate confusion matrix

    :param benchmarking_episodes: Number of episodes that each benchmarking process will run for
    :param createNewEnvironment: OpenAI gym environment creation function
    :param benchmark_job: precomputed BenchmarkingJob
    :param pool: ProcessPoolExecutor shared between benchmarking_jobs to carry out benchmarking matches
    :param matrix_queue: Queue reference sent to benchmarking process, where it will put the bencharmking result
    :param name: BenchmarkingJob name identifier
    """
    benchmark_process = Process(target=benchmark_match_play_process,
                                args=(benchmarking_episodes, createNewEnvironment,
                                      benchmark_job, pool, matrix_queue, name))
    benchmark_process.start()
    return benchmark_process
