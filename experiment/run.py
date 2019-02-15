import os
import sys
sys.path.append(os.path.abspath('..'))

import shutil
import time

from plot_util import create_plots

import yaml
from docopt import docopt

import logging
import logging.handlers
import logging_server

from threading import Thread

import experiment


def initialize_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    socketHandler = logging.handlers.SocketHandler(host='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    logger.addHandler(socketHandler)
    return logger


if __name__ == '__main__':
    import torch
    torch.multiprocessing.set_start_method('forkserver')

    logger = initialize_logger()

    print('''
88888888888888888888888888888888888888888888888888888888O88888888888888888888888
88888888888888888888888888888888888888888888888888888888888O88888888888888888888
8888888888888888888888888888888888888888888888888888888888888O888888888888888888
888888888888888888888888888888888888888888888888888888888888888O8888888888888888
888OZOO88OND88888888888888888888888888888888888888888888888888888O88D88D88888888
888888888D..D8OZO8888888 ....... D88888888888888.........:8888888888D..DO8888888
8888888888DD888888888D..$OOO8888~ .D888888888D...DD88888D,..888888888DD888888888
88888888888888888888Z..O888888888ZZ8OOO888888 .D8888888888D88888888888888OO88888
8888888888..88888888..8888888888888888888888:.OOO88888888888888888888..88888O888
8888888888..8888888$.88888888888888888888888 .88888888888OZO888888888..8888888O8
8888888888..8888888=.888888888,,,,,,,D888888. 88888888,,,,,,,88888OZ8..888888888
8888888888..8888888D.?8888888D88888.+8888888..88888888O8888:.O8888888..888888OOO
8888888888..88888888..D88888888888. 88888888O.:88888888888D..88888888..888888888
8888888888..888888888,..D8888888O .8888888888O..N8888888OD..888888888..8888OO888
8888888888..88888888888..,.?8O... 888888888888OO...,OO=...O8888888888..888888888
8888888888O8888888888888D88I:=O888888888888888888D88~~O888888888O8888O8888888888
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
      --checkpoint_at_iterations INTEGER...   Iteration numbers at which agents will be benchmarked against one another
      --benchmarking_episodes INTEGER         Number of head to head matches used to infer winrates between agents
      --self_play_training_schemes STRING...  Self play training schemes used to choose opponent agent agents during training
      --algorithms STRING...                  Algorithms used to learn a agent
      --fixed_agents STRING...                Fixed agents used to benchmark training agents against
    '''

    options = docopt(_USAGE)
    print(options)

    experiment_id = options['--experiment_id']
    number_of_runs = int(options['--number_of_runs'])

    # TODO create directory structure function
    experiment_directory = 'experiment-{}'.format(experiment_id)

    if os.path.exists(experiment_directory): shutil.rmtree(experiment_directory)
    os.mkdir(experiment_directory)

    t = Thread(target=logging_server.serve_logging_server_forever,
               args=(f'{experiment_directory}/logs',),
               daemon=True)
    t.start()

    with open('{}/experiment_parameters.yml'.format(experiment_directory), 'w') as outfile:
        yaml.dump(options, outfile, default_flow_style=False)

    experiment_durations = []
    for run_id in range(number_of_runs):
        logger.info(f'Starting run: {run_id}')
        start_time = time.time()
        experiment.run_experiment(experiment_id, experiment_directory, run_id, options)
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

    logger.info('Started plot creation')
    create_plots(experiment_directory=experiment_directory, number_of_runs=number_of_runs)
    logger.info('Plots created')

    import signal
    os.kill(os.getpid(), signal.SIGHUP)
