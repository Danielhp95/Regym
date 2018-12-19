import os
import logging

from queue import Empty
import numpy as np


"""
How to read confusion matrix TODO:
    - m[i][j] states the (mean wins, standard deviations) of agent i against agent j
    - this means it's NOT symmetric
"""


# TODO come up with better name
def confusion_matrix_process(training_jobs, checkpoint_iteration_indices, matrix_queue):
    """
    :param training_jobs: Array of TrainingJob namedtuple used to calculate size of confusion matrices
    :param checkpoint_iteration_indices: Array of indices used for choosing which matrix to add stats to
    :param matrix_queue: Queue from which benchmarking stats will be polled
    """
    logger = logging.getLogger('ConfusionMatrixPopulate')
    logger.setLevel(logging.INFO)
    logger.info('Started')

    hashing_dictionary, confusion_matrix_dict = create_confusion_matrix_dictionary(training_jobs, checkpoint_iteration_indices)
    while True:
        # poll for new
        benchmark_statistics = matrix_queue.get()

        # TODO find a better way of doing this. May need to restructure information that is sent around
        logger.info('Received: it:{} ({},{})-({},{})'.format(benchmark_statistics.iteration,
                                                             benchmark_statistics.recorded_policy_vector[0].training_scheme.name,
                                                             benchmark_statistics.recorded_policy_vector[0].policy.name,
                                                             benchmark_statistics.recorded_policy_vector[1].training_scheme.name,
                                                             benchmark_statistics.recorded_policy_vector[1].policy.name))
        logger.debug('Stats: winrates: {} std: {}'.format(benchmark_statistics.winrates, benchmark_statistics.standard_deviations))
        populate_new_statistics(benchmark_statistics, confusion_matrix_dict, hashing_dictionary)
        if check_for_termination(confusion_matrix_dict):
            save_directory = 'confusion_matrices'
            logger.info('All confusion matrices completed. Writing to memory')
            write_matrices(directory=save_directory, matrix_dict=confusion_matrix_dict)
            logger.info('Writing completed')
            break


def create_confusion_matrix_dictionary(training_jobs, checkpoint_iteration_indices):
    hashing_dictionary = {training_job.name: i for i, training_job in enumerate(training_jobs)}
    num_indexes = len(hashing_dictionary)
    empty_confusion_matrix = lambda length: [[None for _ in range(length)] for _ in range(length)]
    return hashing_dictionary, {iteration: empty_confusion_matrix(num_indexes)
                                for iteration in checkpoint_iteration_indices}


def populate_new_statistics(benchmark_stat, confusion_matrix_dict, hashing_dictionary):
    iteration = benchmark_stat.iteration
    index1, index2 = find_indexes(benchmark_stat, hashing_dictionary)
    winrates, standard_devs = benchmark_stat.winrates, benchmark_stat.standard_deviations

    if confusion_matrix_dict[iteration][index1][index2] is not None:
        raise LookupError('Tried to access already populated index: [{},{}]'.format(index1, index2))

    confusion_matrix_dict[iteration][index1][index2] = (winrates[0], standard_devs[0])
    confusion_matrix_dict[iteration][index2][index1] = (winrates[1], standard_devs[1])


def find_indexes(benchmark_stat, hashing_dictionary):
    name_ids = ['{}-{}'.format(rec_policy.training_scheme.name, rec_policy.policy.name)
                for rec_policy in benchmark_stat.recorded_policy_vector]
    return hashing_dictionary[name_ids[0]], hashing_dictionary[name_ids[1]]


def write_matrices(directory, matrix_dict):
    if not os.path.exists(directory):
        os.mkdir(directory)
    # TODO save arrays as np arrays
    for iteration, matrix in matrix_dict.items():
        matrixx = np.array(matrix)
        np.savetxt(f'{directory}/winrates-{iteration}.txt', matrixx[:, :, 0])
        np.savetxt(f'{directory}/std-{iteration}.txt', matrixx[:, :, 1])


def check_for_termination(matrix_dic):
    """
    TODO
    Checks if all matrices have been filled.
    Signaling the end of the process
    :param matrix: Dictionary of confusion matrices
    """
    for matrix in matrix_dic.values():
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] is None:
                    return False
    return True
