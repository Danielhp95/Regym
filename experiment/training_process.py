import os
import sys

sys.path.append(os.path.abspath('..'))

import logging
from multiprocessing import Process

from multiagent_loops.simultaneous_action_rl_loop import self_play_training


def training_process(env, training_policy, self_play_scheme, checkpoint_at_iterations, policy_queue, process_name):
    logger = logging.getLogger(process_name)
    logger.setLevel(logging.DEBUG)
    logger.info('Started')

    completed_iterations = 0
    menagerie = []
    for target_iteration in sorted(checkpoint_at_iterations):
        next_training_iterations = target_iteration - completed_iterations
        menagerie, trained_policy = self_play_training(env=env, training_policy=training_policy,
                                                       self_play_scheme=self_play_scheme, target_episodes=next_training_iterations,
                                                       menagerie=menagerie)
        completed_iterations += target_iteration
        if target_iteration in checkpoint_at_iterations:
            logger.info('Submitted policy at iteration {}'.format(target_iteration))
            policy_queue.put([target_iteration, self_play_scheme, trained_policy])


def create_training_processes(training_jobs, createNewEnvironment, checkpoint_at_iterations, policy_queue):
    """
    :param training_jobs: Array of TrainingJob namedtuples containing a training-scheme, algorithm and name
    :param createNewEnvironment OpenAI gym environment creation function
    :param checkpoint_at_iterations: array containing the episodes at which the policies will be cloned for benchmarking against one another
    :param queue: queue shared among processes to submit policies that will be benchmarked
    """
    logger = logging.getLogger('CreateTrainingProcesses')
    logger.setLevel(logging.DEBUG)

    logger.info('Training {} jobs: [{}]. '.format(len(training_jobs), ', '.join(map(lambda job: job.name, training_jobs))))
    ps = []
    for job in training_jobs:
        p = Process(target=training_process,
                    args=(createNewEnvironment(), job.algorithm, job.training_scheme,
                          checkpoint_at_iterations, policy_queue, job.name))
        p.start()
        ps.append(p) 
    logger.info("All training jobs submitted")
    return ps
