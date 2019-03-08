import os
import sys
#1000, 2000, 3000, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 16]
sys.path.append(os.path.abspath('..'))

import time
import logging
import logging.handlers
from torch.multiprocessing import Process

from rl_algorithms import AgentHook

from multiagent_loops.simultaneous_action_rl_loop import self_play_training


def training_process(envCreationFunction, training_agent, self_play_scheme, checkpoint_at_iterations, agent_queue, process_name, base_path):
    """
    :param envCreationFunction: ParallelEnvironmentCreation wrapper around an environment where agents will be trained on.
    :param training_agent: agent representation + training algorithm which will be trained in this process
    :param self_play_scheme: self play scheme used to meta train the param training_agent.
    :param checkpoint_at_iterations: array containing the episodes at which the agents will be cloned for benchmarking against one another
    :param agent_queue: queue shared among processes to submit agents that will be benchmarked
    :param process_name: String name identifier
    :param base_path: Base directory from where subdirectories will be accessed to reach menageries, save episodic rewards and save checkpoints of agents.
    """
    logger = logging.getLogger(process_name)
    logger.setLevel(logging.DEBUG)
    logger.info('Started')
    logger.addHandler(logging.handlers.SocketHandler(host='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT))

    env = envCreationFunction()

    trained_policy_save_directory = f'{base_path}/{process_name}'
    if not os.path.exists(trained_policy_save_directory):
        os.mkdir(trained_policy_save_directory)

    process_start_time = time.time()

    completed_iterations = 0
    menagerie = []
    training_agent = AgentHook.unhook(training_agent)
    for target_iteration in sorted(checkpoint_at_iterations):
        next_training_iterations = target_iteration - completed_iterations

        training_start = time.time()
        (menagerie, trained_agent,
         trajectories) = self_play_training(env=env, training_agent=training_agent, self_play_scheme=self_play_scheme,
                                            target_episodes=next_training_iterations, iteration=completed_iterations,
                                            menagerie=menagerie, menagerie_path=f'{base_path}/menageries')

        training_duration = time.time() - training_start

        completed_iterations += next_training_iterations

        save_path = f'{trained_policy_save_directory}/{target_iteration}_iterations.pt'
        logger.info(f'Submitted agent at iteration {target_iteration} :: saving at {save_path}')
        hooked_agent = AgentHook(trained_agent.clone(training=False), save_path=save_path)
        agent_queue.put([target_iteration, self_play_scheme, hooked_agent])

        logger.info('Training duration between iterations [{},{}]: {} (seconds)'.format(target_iteration - next_training_iterations, target_iteration, training_duration))

        file_name = '{}-{}.txt'.format(self_play_scheme.name, training_agent.name)
        enumerated_trajectories = zip(range(target_iteration - next_training_iterations, target_iteration), trajectories)
        write_episodic_reward(enumerated_trajectories, target_file_path='{}/episodic_rewards/{}'.format(base_path, file_name))

        # Updating:
        training_agent = trained_agent

    env.close()

    logger.info('All training completed. Total duration: {} seconds'.format(time.time() - process_start_time))
    agent_queue.join()


def write_episodic_reward(enumerated_trajectories, target_file_path):
    with open(target_file_path, 'a') as f:
        for iteration, trajectory in enumerated_trajectories:
            player_1_average_reward = sum(map(lambda t: t[2][0], trajectory)) / len(trajectory) # TODO find a way of not hardcoding indexes
            f.write('{}, {}\n'.format(iteration, player_1_average_reward))


def create_training_processes(training_jobs, training_environment_creation_functions, checkpoint_at_iterations, agent_queue, results_path):
    """
    :param training_jobs: Array of TrainingJob namedtuples containing a training-scheme, algorithm and name
    :param training_environment_creation_functions: ParallelEnvironmentCreationFunction wrapper around an OpenAI gym environment.
    :param checkpoint_at_iterations: array containing the episodes at which the agents will be cloned for benchmarking against one another
    :param agent_queue: queue shared among processes to submit agents that will be benchmarked
    :returns: array of process handlers, needed to join processes at the end of experiment computation
    """
    episodic_reward_directory = f'{results_path}/episodic_rewards'
    if not os.path.exists(episodic_reward_directory):
        os.mkdir(episodic_reward_directory)

    logger = logging.getLogger('CreateTrainingProcesses')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.handlers.SocketHandler(host='localhost', port=logging.handlers.DEFAULT_TCP_LOGGING_PORT))
    logger.info('Training {} jobs: [{}]. '.format(len(training_jobs), ', '.join(map(lambda job: job.name, training_jobs))))
    ps = []
    for job, envCreationFunction in zip(training_jobs, training_environment_creation_functions):
        p = Process(target=training_process,
                    args=(envCreationFunction, job.agent, job.training_scheme,
                          checkpoint_at_iterations, agent_queue, job.name, results_path))
        ps.append(p)
    logger.info("All training jobs submitted")
    return ps
