import os
import sys
sys.path.append(os.path.abspath('..'))

from training_schemes import EmptySelfPlay
from training_schemes import NaiveSelfPlay
from training_schemes import HalfHistorySelfPlay
from training_schemes import FullHistorySelfPlay
from rl_algorithms import TabularQLearning

from training_process import create_training_processes
from match_making import match_making_process
from confusion_matrix_populate_process import confusion_matrix_process


import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from collections import namedtuple
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor

import gym
import gym_rock_paper_scissors
from gym_rock_paper_scissors.fixed_agents import rockAgent, paperAgent


def enumerate_training_jobs(training_schemes, algorithms):
    TrainingJob = namedtuple('TrainingJob', 'training_scheme algorithm name')
    return [TrainingJob(training_scheme, algorithm.clone(training=True), '{}-{}'.format(training_scheme.name, algorithm.name)) for training_scheme in training_schemes for algorithm in algorithms]


if __name__ == '__main__':
    # Initialization
    # Environment
    createNewEnvironment = lambda: gym.make('RockPaperScissors-v0')
    env = createNewEnvironment()

    # Training jobs
    # Throw Rock agent vs Paper agent and see if there's the right amount of wins / std deviation
    training_schemes = [NaiveSelfPlay, FullHistorySelfPlay, HalfHistorySelfPlay]
    algorithms = [TabularQLearning(env.state_space_size, env.action_space_size, env.hash_state)]
    checkpoint_at_iterations = [100] # TODO breaks with more than one iteration
    benchmarking_episodes = 100

    # Performance variables
    benchmark_process_number_workers = 4

    training_jobs = enumerate_training_jobs(training_schemes, algorithms)
    existing_fixed_agents = [] # TODO breaks with fixed agents

    # Queues to communicate processes
    policy_queue = Queue()
    matrix_queue = Queue()

    # Pre processing: Adding fixed agents
    initial_fixed_policies_to_benchmark = [[iteration, EmptySelfPlay, agent]
                                           for agent in existing_fixed_agents
                                           for iteration in checkpoint_at_iterations]
    fixed_policies_for_confusion = enumerate_training_jobs([EmptySelfPlay], existing_fixed_agents) # TODO GET RID OF THIS, it hurts Rewrite bug may come from here
    list(map(policy_queue.put, initial_fixed_policies_to_benchmark)) # Wow, turns out that Python3 requires a conversion to list to force map execution

    expected_number_of_policies = len(training_jobs)*len(checkpoint_at_iterations) + len(initial_fixed_policies_to_benchmark) # TODO make this nicer to calculate

    # Benchmarking
    # Set magic number to number of available cores - (training processes - matchmaking - confusion matrix)
    benchmark_process_pool = ProcessPoolExecutor(max_workers=benchmark_process_number_workers)

    # Create Processes
    training_processes = create_training_processes(training_jobs, createNewEnvironment,
                                                   checkpoint_at_iterations=checkpoint_at_iterations,
                                                   policy_queue=policy_queue)
    mm_process = Process(target=match_making_process,
                         args=(expected_number_of_policies, benchmarking_episodes, createNewEnvironment,
                               policy_queue, matrix_queue, benchmark_process_pool))
    mm_process.start()

    cfm_process = Process(target=confusion_matrix_process,
                          args=(training_jobs + fixed_policies_for_confusion, checkpoint_at_iterations,
                                matrix_queue))
    cfm_process.start()

    # I still need to join on
    cfm_process.join()
    mm_process.join()
    [p.join() for p in training_processes]
