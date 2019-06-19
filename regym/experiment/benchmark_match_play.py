import os
import signal

import time
import logging
import logging.handlers
import random
import numpy as np
import torch
from collections import namedtuple

from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor

from rl_algorithms import AgentHook
from rl_loops.multiagent_loops.simultaneous_action_rl_loop import run_episode

BenchMarkStatistics = namedtuple('BenchMarkStatistics', 'iteration recorded_agent_vector winrates')


def benchmark_match_play_process(expected_benchmarking_matches, benchmarking_episodes, createNewEnvironment, benchmark_queue, matrix_queue, seed):
    """
    :param expected_benchmarking_matches: Number of agents that the process will wait for before shuting itself down
    :param benchmarking_episodes: Number of episodes that each benchmarking process will run for to collect statistics
    :param createNewEnvironment OpenAI gym environment creation function
    :param benchmark_queue: Queue from where BenchmarkingJob(s) will be recieved
    :param matrix_queue: Queue to which submit stats
    """
    logger = logging.getLogger('Benchmarking')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.handlers.SocketHandler(host='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT))
    logger.info('Started')

    np.random.seed(seed)
    torch.manual_seed(seed)

    received_agents = 0
    while True:
        benchmark_job = benchmark_queue.get()
        received_agents += 1
        benchmark_queue.task_done()
        logger.info('Received {}. {}/{} received. Started for {} episodes'.format(benchmark_job.name, received_agents, expected_benchmarking_matches, benchmarking_episodes))

        agent_vector = [recorded_agent.agent for recorded_agent in benchmark_job.recorded_agent_vector]

        winrates = benchmark_empirical_winrates(benchmarking_episodes, createNewEnvironment, agent_vector, logger)

        matrix_queue.put(BenchMarkStatistics(benchmark_job.iteration,
                                             benchmark_job.recorded_agent_vector,
                                             winrates))
        check_for_termination(received_agents, expected_benchmarking_matches, matrix_queue, logger)


def benchmark_empirical_winrates(benchmarking_episodes, createNewEnvironment, agent_vector, logger):
    with ProcessPoolExecutor(max_workers=3) as executor:
        benchmark_start = time.time()
        futures = [executor.submit(single_match, *[createNewEnvironment(), agent_vector])
                   for _ in range(benchmarking_episodes)]

        wins_vector = np.zeros(len(agent_vector))

        for future in as_completed(futures):
            episode_winner = future.result()
            wins_vector[episode_winner] += 1
        benchmark_duration = time.time() - benchmark_start
    logger.info('Benchmarking finished. Duration: {} seconds'.format(benchmark_duration))
    winrates = wins_vector / benchmarking_episodes
    return winrates


def single_match(createNewEnvironment, agent_vector):
    # trajectory: [(s,a,r,s')]
    unhooked_agents = [AgentHook.unhook(agent, use_cuda=False) for agent in agent_vector]
    trajectory = run_episode(env, unhooked_agents, training=False)
    #trajectory = run_episode(createNewEnvironment(), unhooked_agents, training=False)
    reward_vector = lambda t: t[2]
    individal_agent_trajectory_reward = lambda t, agent_index: sum(map(lambda experience: reward_vector(experience)[agent_index], t))
    cumulative_reward_vector = [individal_agent_trajectory_reward(trajectory, i) for i in range(len(agent_vector))]
    episode_winner = choose_winner(cumulative_reward_vector)
    return episode_winner


def choose_winner(cumulative_reward_vector, break_ties=random.choice):
    indexes_max_score = np.argwhere(cumulative_reward_vector == np.amax(cumulative_reward_vector))
    return break_ties(indexes_max_score.flatten().tolist())


def check_for_termination(received_agents, expected_number_of_agents, matrix_queue, logger):
    """
    Checks if process should be killed because all processing has been submitted.
    That is, all expected agents have been received and all benchmarking
    child processes have been created.

    :param received_agents: Number of agents received so far
    :param expected_number_of_agents: Number of agents that the process will wait for before shuting itself down
    """
    if received_agents >= expected_number_of_agents:
        logger.info('All expected trained agents have been recieved. Shutting down')
        matrix_queue.join()
        os.kill(os.getpid(), signal.SIGTERM)
