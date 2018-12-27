import os
from os import listdir
from os.path import isfile, join

import numpy as np

import matplotlib
from matplotlib import pyplot as plt


# TODO Get confusion matrices and plot their results using heatmaps. Could be useful using pandas for this instead of numpy arrays.
def create_plots(experiment_directory, number_of_runs):
    for run_id in range(number_of_runs):
        os.mkdir(f'{experiment_directory}/run-{run_id}/plots')
        plot_single_run(run_id=run_id,
                        source_dir=f'{experiment_directory}/run-{run_id}')


def plot_single_run(run_id, source_dir):
    create_confusion_matrix_heatmap(source_dir=f'{source_dir}/confusion_matrices',
                                    target_dir=f'{source_dir}/plots')
    plt.close()
    create_average_winrate_graph(source_dir=f'{source_dir}/winrates',
                                 target_dir=f'{source_dir}/plots')
    plt.close()
    create_individual_episodic_reward_graph(source_dir=f'{source_dir}/episodic_rewards',
                                            target_dir=f'{source_dir}/plots')
    plt.close()


def create_confusion_matrix_heatmap(source_dir, target_dir):
    files = all_files_in_directory(source_dir)
    for f in files:
        create_single_matrix(f, target_dir)


def create_single_matrix(source, target_dir):
    file_name = get_file_name_from_full_path(source)
    file_content = np.loadtxt(open(source, 'rb'), delimiter=', ')

    plt.imshow(file_content, cmap='inferno')
    plt.savefig(f'{target_dir}/heatmap-{file_name}.png')


def create_average_winrate_graph(source_dir, target_dir):
    files = all_files_in_directory(source_dir)
    for f in files:
        file_name = get_file_name_from_full_path(f)
        file_content = np.loadtxt(open(f, 'rb'), delimiter=', ')
        iterations   = file_content[:, 0]
        winrates   = file_content[:, 1]

        y_max = 100
        plt.xticks(iterations)
        plt.yticks(np.arange(0, y_max * + 20, 10)) # Watch out when changing y_max
        plt.plot((0, max(iterations)), (y_max / 2, y_max / 2), '--')
        plt.plot(iterations, winrates, marker='o', label=file_name) # TODO Use markers

    plt.ylim(0, 100) # TODO fix winrate and use 0, 1
    plt.legend(loc='best')
    plt.savefig(f'{target_dir}/benchmark_winrates.png')


def create_individual_episodic_reward_graph(source_dir, target_dir):
    for f in all_files_in_directory(source_dir):
        file_name = get_file_name_from_full_path(f)
        training_scheme, algorithm = file_name.split('-')

        file_content = np.loadtxt(open(f, 'rb'), delimiter=', ')
        iterations   = file_content[:, 0]
        avg_reward   = file_content[:, 1]

        plt.xlabel('Training episode')
        plt.ylabel('Average episodic reward')
        plt.plot(iterations, avg_reward)

        plt.savefig(f'{target_dir}/episodic_reward-{file_name}.png')


def all_files_in_directory(directory):
    return [join(directory, f)
            for f in os.listdir(directory) if isfile(join(directory, f))]


def get_file_name_from_full_path(filename):
    return os.path.splitext(filename)[0].split('/')[-1]
