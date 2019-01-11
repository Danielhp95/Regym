import os
import sys
sys.path.append(os.path.abspath('..'))

import shutil
import time

from training_schemes import EmptySelfPlay, NaiveSelfPlay, HalfHistorySelfPlay, FullHistorySelfPlay
from rl_algorithms import TabularQLearning

from plot_util import create_plots

from training_process import create_training_processes
from match_making import match_making_process
from confusion_matrix_populate_process import confusion_matrix_process


import yaml
from docopt import docopt
import logging

# TODO Use an extra queue to receive logging from a a queue,
# or even a socket: https://docs.python.org/3/howto/logging-cookbook.html#sending-and-receiving-logging-events-across-a-network
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from collections import namedtuple
#from multiprocessing import Process, Queue
from torch.multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor

import gym
import gym_rock_paper_scissors
from gym_rock_paper_scissors.fixed_agents import rockAgent, paperAgent, scissorsAgent

TrainingJob = namedtuple('TrainingJob', 'training_scheme algorithm name')


def enumerate_training_jobs(training_schemes, algorithms):
    return [TrainingJob(training_scheme, algorithm.clone(training=True), '{}-{}'.format(training_scheme.name, algorithm.name)) for training_scheme in training_schemes for algorithm in algorithms]


# TODO find better name
def preprocess_fixed_agents(existing_fixed_agents, checkpoint_at_iterations):
    initial_fixed_policies_to_benchmark = [[iteration, EmptySelfPlay, agent]
                                           for agent in existing_fixed_agents
                                           for iteration in checkpoint_at_iterations]
    fixed_policies_for_confusion = enumerate_training_jobs([EmptySelfPlay], existing_fixed_agents) # TODO GET RID OF THIS
    return initial_fixed_policies_to_benchmark, fixed_policies_for_confusion


def create_all_initial_processes(training_jobs, createNewEnvironment, checkpoint_at_iterations,
                                 policy_queue, matrix_queue, benchmarking_episodes,
                                 fixed_policies_for_confusion, results_path):

    # TODO Set magic number to number of available cores - (training processes - matchmaking - confusion matrix)
    benchmark_process_number_workers = 4
    benchmark_process_pool = ProcessPoolExecutor(max_workers=benchmark_process_number_workers)

    training_processes = create_training_processes(training_jobs, createNewEnvironment,
                                                   checkpoint_at_iterations=checkpoint_at_iterations,
                                                   policy_queue=policy_queue, results_path=results_path)

    expected_number_of_policies = (len(training_jobs) + len(fixed_policies_for_confusion)) * len(checkpoint_at_iterations)
    mm_process = Process(target=match_making_process,
                         args=(expected_number_of_policies, benchmarking_episodes, createNewEnvironment,
                               policy_queue, matrix_queue, benchmark_process_pool))

    cfm_process = Process(target=confusion_matrix_process,
                          args=(training_jobs + fixed_policies_for_confusion, checkpoint_at_iterations,
                                matrix_queue, results_path))
    return (training_processes, mm_process, cfm_process)


def define_environment_creation_funcion(environment_name_cli):
    valid_environments = ['RockPaperScissors-v0']
    if environment_name_cli not in valid_environments:
        raise ValueError("Unknown environment {}\t valid environments: {}".format(environment_name_cli, valid_environments))
    return lambda: gym.make(environment_name_cli)


def run_processes(training_processes, mm_process, cfm_process):
    [p.start() for p in training_processes]
    mm_process.start()
    cfm_process.start()

    [p.join() for p in training_processes]
    mm_process.join()
    cfm_process.join()


def initialize_training_schemes(training_schemes_cli):
    def parse_training_scheme(training_scheme):
        if training_scheme.lower() == 'fullhistoryselfplay': return FullHistorySelfPlay
        elif training_scheme.lower() == 'halfhistoryselfplay': return HalfHistorySelfPlay
        elif training_scheme.lower() == 'naiveselfplay': return NaiveSelfPlay
        else: raise ValueError('Unknown training scheme {}. Try defining it inside this script.'.format(training_scheme))
    return [parse_training_scheme(t_s) for t_s in training_schemes_cli]


def initialize_algorithms(environment, algorithms_cli):
    def parse_algorithm(algorithm, env):
        if algorithm.lower() == 'tabularqlearning':
            return TabularQLearning(env.state_space_size, env.action_space_size, env.hash_state)
        if algorithm.lower() == 'deepqlearning':
            from rl_algorithms import build_DQN_Agent
            return build_DQN_Agent(state_space_size=env.state_space_size, action_space_size=env.action_space_size, hash_function=env.hash_state, double=True, dueling=True)
        else: raise ValueError('Unknown algorithm {}. Try defining it inside this script.'.format(algorithm))
    return [parse_algorithm(algorithm, environment) for algorithm in algorithms_cli]


def initialize_fixed_agents(fixed_agents_cli):
    def parse_fixed_agent(agent):
        if agent.lower() == 'rockagent': return rockAgent
        elif agent.lower() == 'paperagent': return paperAgent
        elif agent.lower() == 'scissorsagent': return scissorsAgent
        else: raise ValueError('Unknown fixed agent {}. Try defining it inside this script.'.format(agent))
    return [parse_fixed_agent(agent) for agent in fixed_agents_cli]


def run_experiment(experiment_id, experiment_directory, number_of_runs, options, logger):
    createNewEnvironment  = define_environment_creation_funcion(options['--environment'])
    env = createNewEnvironment()

    checkpoint_at_iterations = [int(i) for i in options['--checkpoint_at_iterations'].split(',')]
    benchmarking_episodes    = int(options['--benchmarking_episodes'])

    training_schemes = initialize_training_schemes(options['--self_play_training_schemes'].split(','))
    algorithms       = initialize_algorithms(env, options['--algorithms'].split(','))
    fixed_agents     = initialize_fixed_agents(options['--fixed_agents'].split(','))

    training_jobs = enumerate_training_jobs(training_schemes, algorithms)

    (initial_fixed_policies_to_benchmark,
     fixed_policies_for_confusion) = preprocess_fixed_agents(fixed_agents, checkpoint_at_iterations)
    policy_queue, matrix_queue = Queue(), Queue()

    logger.info(f'Starting run: {run_id}')
    results_path = f'{experiment_directory}/run-{run_id}'
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    list(map(policy_queue.put, initial_fixed_policies_to_benchmark)) # Add initial fixed policies to be benchmarked

    (training_processes,
     mm_process,
     cfm_process) = create_all_initial_processes(training_jobs, createNewEnvironment, checkpoint_at_iterations,
                                                 policy_queue, matrix_queue, benchmarking_episodes,
                                                 fixed_policies_for_confusion, results_path)

    run_processes(training_processes, mm_process, cfm_process)


if __name__ == '__main__':
    import torch
    torch.multiprocessing.set_start_method('spawn')
    logger.info('''
88888888888888888888888888888888888888888888888888888888O88888888888888888888888
88888888888888888888888888888888888888888888888888888888888O88888888888888888888
8888888888888888888888888888888888888888888888888888888888888O888888888888888888
888888888888888888888888888888888888888888888888888888888888888O8888888888888888
888OZOO88OND88888888888888888888888888888888888888888888888888888O88D88D88888888
888888888D..D8OZO8888888 ....... D88888888888888.........:8888888888...O88888888
8888888888DD888888888D..$OOO8888~ .D888888888D...DD88888D,..88888888O8O888888888
88888888888888888888Z..O888888888ZZ8OOO888888 .D8888888888D8888888888888OO888888
8888888888..88888888..8888888888888888888888:.OOO88888888888888888888.88888O8888
8888888888..8888888$.88888888888888888888888 .88888888888OZO888888888.8888888O88
8888888888..8888888=.888888888,,,,,,,D888888. 88888888,,,,,,,88888OZO.8888888888
8888888888..8888888D.?8888888D88888.+8888888..88888888O8888:.O8888888.888888OOOO
8888888888..88888888..D88888888888. 88888888O.:88888888888D..88888888.8888888888
8888888888..888888888,..D8888888O .8888888888O..N8888888OD..888888888.8888OO8888
8888888888..88888888888..,.?8O... 888888888888OO...,OO=...O8888888888.8888888888
8888888888O8888888888888D88I:=O888888888888888888D88~~O888888888O88888O888888888
88888888888888888888888888888888888888888888888888888888888OO8888888888888888888
888888888888888888888888888888888888888888888888888888O8888888888888888888888888
888888888888888888888888888888888888888888888888OO888888888888888888888888888888
8888888888888888888888888888888888888888888OO88888888888888888888888888888888888
''')

    _USAGE = '''
    Usage:
      run [options]

    Options:
      --environment STRING                    OpenAI environment used to train agents on
      --experiment_id STRING                  Experimment id used to identify between different experiments
      --number_of_runs INTEGER                Number of runs used to calculate standard deviations for various metrics
      --checkpoint_at_iterations INTEGER...   Iteration numbers at which policies will be benchmarked against one another
      --benchmarking_episodes INTEGER         Number of head to head matches used to infer winrates between policies
      --self_play_training_schemes STRING...  Self play training schemes used to choose opponent agent policies during training
      --algorithms STRING...                  Algorithms used to learn a policy
      --fixed_agents STRING...                Fixed agents used to benchmark training policies against
    '''

    options = docopt(_USAGE)
    logger.info(options)

    experiment_id = int(options['--experiment_id'])
    number_of_runs = int(options['--number_of_runs'])

    # TODO create directory structure function
    experiment_directory = 'experiment-{}'.format(experiment_id)
    if os.path.exists(experiment_directory): shutil.rmtree(experiment_directory)
    os.mkdir(experiment_directory)

    with open('{}/experiment_parameters.yml'.format(experiment_directory), 'w') as outfile:
        yaml.dump(options, outfile, default_flow_style=False)

    experiment_durations = []
    for run_id in range(number_of_runs):
        start_time = time.time()
        run_experiment(experiment_id, experiment_directory, number_of_runs, options, logger)
        experiment_duration = time.time() - start_time
        experiment_durations.append(experiment_duration)
        logger.info('Finished run: {}. Duration: {} (seconds)\n'.format(run_id, experiment_duration))

    import numpy as np
    total_experiment_duration = sum(experiment_durations)
    average_experiment_duration = np.mean(experiment_durations)
    standard_deviation_experiment_duration = np.std(experiment_durations)

    logger.info('Total experiment duration: {}'.format(total_experiment_duration))
    logger.info('Experiment mean run duration: {}'.format(average_experiment_duration))
    logger.info('Experiment std dev duration:  {}'.format(standard_deviation_experiment_duration))

    create_plots(experiment_directory=experiment_directory, number_of_runs=number_of_runs)
