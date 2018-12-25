import os
import sys
sys.path.append(os.path.abspath('..'))

import shutil

from training_schemes import EmptySelfPlay, NaiveSelfPlay, HalfHistorySelfPlay, FullHistorySelfPlay
from rl_algorithms import TabularQLearning

from plot_util import create_plots

from training_process import create_training_processes
from match_making import match_making_process
from confusion_matrix_populate_process import confusion_matrix_process


import logging

# TODO Use an extra queue to receive logging from a a queue,
# or even a socket: https://docs.python.org/3/howto/logging-cookbook.html#sending-and-receiving-logging-events-across-a-network
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from collections import namedtuple
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor

import gym
import gym_rock_paper_scissors
from gym_rock_paper_scissors.fixed_agents import rockAgent, paperAgent

TrainingJob = namedtuple('TrainingJob', 'training_scheme algorithm name')


def enumerate_training_jobs(training_schemes, algorithms):
    return [TrainingJob(training_scheme, algorithm.clone(training=True), '{}-{}'.format(training_scheme.name, algorithm.name)) for training_scheme in training_schemes for algorithm in algorithms]


# TODO find better name
def preprocess_fixed_agents(existing_fixed_agents, checkpoint_at_iterations):
    initial_fixed_policies_to_benchmark = [[iteration, EmptySelfPlay, agent]
                                           for agent in fixed_agents
                                           for iteration in checkpoint_at_iterations]
    fixed_policies_for_confusion = enumerate_training_jobs([EmptySelfPlay], fixed_agents) # TODO GET RID OF THIS, it hurts Rewrite bug may come from here
    return initial_fixed_policies_to_benchmark, fixed_policies_for_confusion


def create_all_initial_processes(training_jobs, createNewEnvironment, checkpoint_at_iterations,
                                 policy_queue, matrix_queue, benchmarking_episodes,
                                 fixed_policies_for_confusion, results_path):

    # Set magic number to number of available cores - (training processes - matchmaking - confusion matrix)
    benchmark_process_number_workers = 4
    benchmark_process_pool = ProcessPoolExecutor(max_workers=benchmark_process_number_workers)

    training_processes = create_training_processes(training_jobs, createNewEnvironment,
                                                   checkpoint_at_iterations=checkpoint_at_iterations,
                                                   policy_queue=policy_queue)

    expected_number_of_policies = (len(training_jobs) + len(fixed_policies_for_confusion)) * len(checkpoint_at_iterations)
    mm_process = Process(target=match_making_process,
                         args=(expected_number_of_policies, benchmarking_episodes, createNewEnvironment,
                               policy_queue, matrix_queue, benchmark_process_pool))

    cfm_process = Process(target=confusion_matrix_process,
                          args=(training_jobs + fixed_policies_for_confusion, checkpoint_at_iterations,
                                matrix_queue, results_path))
    return (training_processes, mm_process, cfm_process)


def define_environment_creation_funcion():
    return lambda: gym.make('RockPaperScissors-v0')


def create_interprocess_queues():
    return (Queue(), Queue())


def run_processes(training_process, mm_process, cfm_process):
    [p.start() for p in training_processes]
    mm_process.start()
    cfm_process.start()

    cfm_process.join()
    mm_process.join()
    [p.join() for p in training_processes]


def initialize_algorithms(environment):
    algorithms = [TabularQLearning(env.state_space_size, env.action_space_size, env.hash_state)]
    return algorithms


def initialize_fixed_agents():
    # return [rockAgent, paperAgent]
    return [rockAgent]


if __name__ == '__main__':
    createNewEnvironment  = define_environment_creation_funcion()
    env = createNewEnvironment()

    experiment_id = 0 # TODO make this into script param
    number_of_runs = 1 # TODO make this into script param

    checkpoint_at_iterations = [100, 1000]
    benchmarking_episodes    = 100

    training_schemes = [NaiveSelfPlay] # , FullHistorySelfPlay] # , FullHistorySelfPlay, HalfHistorySelfPlay]
    algorithms       = initialize_algorithms(env)
    fixed_agents     = initialize_fixed_agents()

    training_jobs = enumerate_training_jobs(training_schemes, algorithms)

    policy_queue, matrix_queue = create_interprocess_queues()

    (initial_fixed_policies_to_benchmark,
     fixed_policies_for_confusion) = preprocess_fixed_agents(fixed_agents, checkpoint_at_iterations)

    # Remove existing experiment
    if os.path.exists(str(experiment_id)): shutil.rmtree(str(experiment_id))
    os.mkdir(str(experiment_id))

    for run_id in range(number_of_runs):
        logger.info(f'Starting run: {run_id}')
        results_path = f'{experiment_id}/run-{run_id}'
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        list(map(policy_queue.put, initial_fixed_policies_to_benchmark)) # Add initial fixed policies to be benchmarked

        (training_processes,
         mm_process,
         cfm_process) = create_all_initial_processes(training_jobs, createNewEnvironment, checkpoint_at_iterations,
                                                     policy_queue, matrix_queue, benchmarking_episodes,
                                                     fixed_policies_for_confusion, results_path)

        run_processes(training_processes, mm_process, cfm_process)
        logger.info(f'Finished run: {run_id}\n')

    create_plots(experiment_directory=str(experiment_id), number_of_runs=number_of_runs)
