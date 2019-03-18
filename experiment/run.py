import os
import sys
sys.path.append(os.path.abspath('..'))

import shutil
import time
import numpy as np
import yaml
from docopt import docopt
from threading import Thread

import logging
import logging.handlers
import logging_server

from util.experiment_parsing import filter_relevant_agent_configurations
import experiment
from plot_util import create_plots


def initialize_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    socketHandler = logging.handlers.SocketHandler(host='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT)
    logger.addHandler(socketHandler)
    return logger


if __name__ == '__main__':
    import torch
    torch.multiprocessing.set_start_method('forkserver')
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1])) # Not impressed

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
        --config String   Path to Yaml experiment configuration file [default: ./experiment_config.yaml]
        --dest String     Path where experiment output will be stored [default: ./]
    '''

    docopt_options = docopt(_USAGE)
    print(docopt_options)
    all_configs = yaml.load(open(docopt_options['--config']), Loader=yaml.FullLoader)
    experiment_config = all_configs['experiment']
    relevant_agent_configuration = filter_relevant_agent_configurations(experiment_config, all_configs['agents'])

    experiment_path = docopt_options['--dest']
    experiment_id   = experiment_config['experiment_id']
    number_of_runs  = int(experiment_config['number_of_runs'])
    seeds = list(map(int, experiment_config['seeds'])) if 'seeds' in experiment_config else np.random.randint(0, 10000, number_of_runs).tolist()
    if len(seeds) < number_of_runs:
        print(f'Number of random seeds does not match "number of runs" config value. Genereting new seeds"')
        seeds = np.random.randint(0, 10000, number_of_runs)
    experiment_config['seeds'] = seeds

    experiment_directory = f'{experiment_path}/experiment-{experiment_id}'

    if os.path.exists(experiment_directory): shutil.rmtree(experiment_directory)
    os.makedirs(experiment_directory, exist_ok=True)

    t = Thread(target=logging_server.serve_logging_server_forever,
               args=(f'{experiment_directory}/logs',),
               daemon=True)
    t.start()

    all_relevant_config = {'experiment': experiment_config, 'agents': relevant_agent_configuration}
    with open('{}/experiment_parameters.yml'.format(experiment_directory), 'w') as outfile:
        yaml.dump(all_relevant_config, outfile, default_flow_style=False)

    experiment_durations = []
    for run_id in range(number_of_runs):
        logger.info(f'Starting run: {run_id}')
        start_time = time.time()
        experiment.run_experiment(experiment_id, experiment_directory, run_id, experiment_config, relevant_agent_configuration, seeds[run_id])
        experiment_duration = time.time() - start_time
        experiment_durations.append(experiment_duration)
        logger.info('Finished run: {}. Duration: {} (seconds)\n'.format(run_id, experiment_duration))

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
