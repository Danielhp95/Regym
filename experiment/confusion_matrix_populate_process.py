import os
import logging

from queue import Empty
import numpy as np


"""
How to read confusion matrix TODO:
    - m[i][j] states the winrate of agent i against agent j
    - this means it's NOT symmetric
"""


# TODO come up with better name
def confusion_matrix_process(training_jobs, checkpoint_iteration_indices, matrix_queue, results_path):
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
        populate_new_statistics(benchmark_statistics, confusion_matrix_dict, hashing_dictionary)
        if check_for_termination(confusion_matrix_dict):
            logger.info('All confusion matrices completed. Writing to memory')

            filled_matrices = {key: fill_winrate_diagonal(confusion_matrix, value='0.5') for key, confusion_matrix in confusion_matrix_dict.items()}
            write_matrices(directory='{}/confusion_matrices'.format(results_path), matrix_dict=filled_matrices)
            write_average_winrates(directory='{}/winrates'.format(results_path), matrix_dict=filled_matrices, hashing_dictionary=hashing_dictionary)

            write_legend_file(hashing_dictionary, path='{}/confusion_matrices/legend.txt'.format(results_path))
            logger.info('Writing completed')
            break


def fill_winrate_diagonal(matrix, value):
    np.fill_diagonal(matrix, value)
    return matrix


def create_confusion_matrix_dictionary(training_jobs, checkpoint_iteration_indices):
    hashing_dictionary = {training_job.name: i for i, training_job in enumerate(training_jobs)}
    num_indexes = len(hashing_dictionary)

    def filled_matrix(size):
        m = np.empty((size, size))
        m.fill(-1)
        return m

    return hashing_dictionary, {iteration: filled_matrix(num_indexes)
                                for iteration in checkpoint_iteration_indices}


def populate_new_statistics(benchmark_stat, confusion_matrix_dict, hashing_dictionary):
    iteration = benchmark_stat.iteration
    index1, index2 = find_indexes(benchmark_stat, hashing_dictionary)
    winrates = benchmark_stat.winrates

    # if confusion_matrix_dict[iteration][index1][index2] is not None and index1 != index2:
    #     raise LookupError('Tried to access already populated index: [{},{}]'.format(index1, index2))

    confusion_matrix_dict[iteration][index1][index2] = winrates[0]
    confusion_matrix_dict[iteration][index2][index1] = winrates[1]


def find_indexes(benchmark_stat, hashing_dictionary):
    name_ids = ['{}-{}'.format(rec_policy.training_scheme.name, rec_policy.policy.name)
                for rec_policy in benchmark_stat.recorded_policy_vector]
    return hashing_dictionary[name_ids[0]], hashing_dictionary[name_ids[1]]


def write_matrices(directory, matrix_dict):
    if not os.path.exists(directory):
        os.mkdir(directory)
    for iteration, matrix in matrix_dict.items():
        winrate_matrix = np.array(matrix)
        np.savetxt('{}/confusion_matrix-{}.txt'.format(directory,iteration), winrate_matrix[:, :], delimiter=', ')


def write_average_winrates(directory, matrix_dict, hashing_dictionary):
    if not os.path.exists(directory):
        os.mkdir(directory)
    for name, index in hashing_dictionary.items():
        with open('{}/{}.txt'.format(directory,name), 'a') as f:
            for iteration, matrix in matrix_dict.items():
                avg_winrate = sum(matrix[index]) / len(matrix)
                f.write('{}, {}\n'.format(iteration,avg_winrate))


def check_for_termination(matrix_dic):
    """
    Checks if all matrices have been filled.
    Signaling the end of the process
    :param matrix: Dictionary of confusion matrices
    """
    for matrix in matrix_dic.values():
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == -1:
                    return False
    return True


def write_legend_file(hashing_dictionary, path):
    with open(path, 'w') as f:
        for name, index in hashing_dictionary.items():
            f.write('{}, {}\n'.format(name,index))
