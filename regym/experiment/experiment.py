import os
import sys
sys.path.append(os.path.abspath('..'))

from math import factorial
import numpy as np
import torch
from torch.multiprocessing import Process, JoinableQueue
import gym
import gym_rock_paper_scissors
from tensorboardX import SummaryWriter

import util
from training_schemes import EmptySelfPlay, NaiveSelfPlay, HalfHistorySelfPlay, FullHistorySelfPlay, HalfHistoryLimitSelfPlay, HalfHistoryLimitSelfPlay

from rl_algorithms import AgentHook
from environments import ParallelEnv, EnvironmentCreationFunction, ParallelEnvironmentCreationFunction

from training_process import create_training_processes
from match_making import match_making_process
from benchmark_match_play import benchmark_match_play_process
from confusion_matrix_populate_process import confusion_matrix_process

from collections import namedtuple
TrainingJob = namedtuple('TrainingJob', 'training_scheme agent name')


def enumerate_training_jobs(training_schemes, algorithms, env_name):
    a = [TrainingJob(training_scheme, AgentHook(algorithm.clone(training=True)), f'{training_scheme.name}-{algorithm.name}')
        for training_scheme in training_schemes
        for algorithm in algorithms]
    b = [EnvironmentCreationFunction(env_name) if agent.nbr_actor == 1 else ParallelEnvironmentCreationFunction(env_name, agent.nbr_actor)
        for training_scheme in training_schemes
        for agent in algorithms]
    c = EnvironmentCreationFunction(env_name)
    return a,b,c

def preprocess_fixed_agents(existing_fixed_agents, checkpoint_at_iterations, env_name):
    initial_fixed_agents_to_benchmark = [[iteration, EmptySelfPlay, AgentHook(agent)]
                                         for agent in existing_fixed_agents
                                         for iteration in checkpoint_at_iterations]
    fixed_agents_for_confusion, _, _ = enumerate_training_jobs([EmptySelfPlay], existing_fixed_agents, env_name) # TODO GET RID OF THIS
    return initial_fixed_agents_to_benchmark, fixed_agents_for_confusion


def create_all_initial_processes(training_jobs, createNewEnvironment, checkpoint_at_iterations,
                                 agent_queue, benchmark_queue, matrix_queue, benchmarking_episodes,
                                 fixed_agents_for_confusion, results_path, seed):

    total_agents = len(training_jobs) + len(fixed_agents_for_confusion)
    expected_number_of_agents = total_agents * len(checkpoint_at_iterations)
    expected_benchmarking_matches = (int((total_agents * (total_agents - 1)) / 2) + total_agents) * len(checkpoint_at_iterations)
    training_processes = create_training_processes(training_jobs, createNewEnvironment, checkpoint_at_iterations=checkpoint_at_iterations, agent_queue=agent_queue, results_path=results_path, seed=seed)
    mm_process = Process(target=match_making_process, args=(expected_number_of_agents, agent_queue, benchmark_queue))
    benchmark_process = Process(target=benchmark_match_play_process, args=(expected_benchmarking_matches, benchmarking_episodes, createNewEnvironment, benchmark_queue, matrix_queue, seed))
    cfm_process = Process(target=confusion_matrix_process, args=(training_jobs + fixed_agents_for_confusion, checkpoint_at_iterations, matrix_queue, results_path))
    return training_processes, mm_process, benchmark_process, cfm_process



def run_processes(training_processes, mm_process, benchmark_process, cfm_process):
    [p.start() for p in training_processes]
    mm_process.start()
    benchmark_process.start()
    cfm_process.start()

    [p.join() for p in training_processes]
    mm_process.join()
    benchmark_process.join()
    cfm_process.join()


def run_experiment(experiment_id, experiment_directory, run_id, experiment_config, agents_config, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    results_path = f'{experiment_directory}/run-{run_id}'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    createNewEnvironment  = EnvironmentCreationFunction(experiment_config['environment'])

    checkpoint_at_iterations = [int(i) for i in experiment_config['checkpoint_at_iterations']]
    benchmarking_episodes    = int(experiment_config['benchmarking_episodes'])

    training_schemes  = util.experiment_parsing.initialize_training_schemes(experiment_config['self_play_training_schemes'])
    algorithms        = util.experiment_parsing.initialize_algorithms(createNewEnvironment(), agents_config)
    fixed_agents      = util.experiment_parsing.initialize_fixed_agents(experiment_config['fixed_agents'])

    training_jobs, training_environments, benchmarking_environment = enumerate_training_jobs(training_schemes, algorithms, experiment_config['environment'])

    (initial_fixed_agents_to_benchmark, fixed_agents_for_confusion) = preprocess_fixed_agents(fixed_agents, checkpoint_at_iterations)
    agent_queue, benchmark_queue, matrix_queue = JoinableQueue(), JoinableQueue(), JoinableQueue()

    for fixed_agent in initial_fixed_agents_to_benchmark: agent_queue.put(fixed_agent)

    (training_processes,
     mm_process, benchmark_process,
     cfm_process) = create_all_initial_processes(training_jobs, createNewEnvironment, checkpoint_at_iterations,
                                                 agent_queue, benchmark_queue, matrix_queue, benchmarking_episodes,
                                                 fixed_agents_for_confusion, results_path, seed)

    run_processes(training_processes, mm_process, benchmark_process, cfm_process)
