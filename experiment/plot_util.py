import os
from os import listdir
from os.path import isfile, join

import numpy as np

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt


def create_plots(experiment_directory, number_of_runs):
    os.mkdir('{}/plots'.format(experiment_directory))

    aggregated_episodic_reward_dict, aggregated_benchmark_winrate_dict = {}, {}
    for run_id in range(number_of_runs):
        os.mkdir('{}/run-{}/plots'.format(experiment_directory,run_id))
        new_episodic_reward_dict, \
        new_winrate_dict = plot_single_run(run_id=run_id,
                                           source_dir='{}/run-{}'.format(experiment_directory,run_id))
        update_aggregated_statistics(aggregated_episodic_reward_dict, aggregated_benchmark_winrate_dict,
                                     new_episodic_reward_dict, new_winrate_dict)

    create_aggregated_plots(aggregated_episodic_reward_dict,
                            aggregated_benchmark_winrate_dict,
                            target_dir='{}/plots'.format(experiment_directory) )


def update_aggregated_statistics(aggregated_episodic_reward_dict, aggregated_benchmark_winrate_dict,
                                 new_episodic_reward_dict, new_winrate_dict):
    update_episodic_reward(aggregated_episodic_reward_dict, new_episodic_reward_dict)
    update_benchmark_winrates(aggregated_benchmark_winrate_dict, new_winrate_dict)


def update_benchmark_winrates(old_dict, new_dict):
    for name, (iterations, winrates) in new_dict.items():
        if name in old_dict: old_dict[name].append((iterations, winrates))
        if name not in old_dict: old_dict[name] = [(iterations, winrates)]


def update_episodic_reward(old_dict, new_dict):
    for name, episodic_rewards in new_dict.items():
        if name in old_dict: old_dict[name].append(episodic_rewards)
        if name not in old_dict: old_dict[name] = [episodic_rewards]


def create_aggregated_plots(episodic_reward_dict, benchmark_winrate_dict, target_dir):
    create_aggregated_individual_episodic_reward_plots(episodic_reward_dict, target_dir)
    create_aggregated_benchmark_winrate_plot(benchmark_winrate_dict, target_dir)


def create_aggregated_individual_episodic_reward_plots(episodic_reward_dict, target_dir):
    for name, episodic_rewards in episodic_reward_dict.items():
        episodic_reward_matrix = np.array(episodic_rewards) # Make 1D array of np arrays into 2D numpy array
        # plot single aggregated reward
        iterations = np.arange(0, episodic_reward_matrix.shape[1])
        means, standard_deviations = episodic_reward_matrix.mean(axis=0,), episodic_reward_matrix.std(axis=0)
        upper_bound = [mean + std for mean, std in zip(means, standard_deviations)]
        lower_bound = [mean - std for mean, std in zip(means, standard_deviations)]
        plt.plot(iterations, means)
        plt.fill_between(iterations, upper_bound, lower_bound, alpha=0.4)

        plt.xlabel('Training episode')
        plt.ylabel('Average episodic reward')
        plt.title('Average episodic reward \nfor policy: {}'.format(name))

        plt.savefig('{}/episodic_reward-{}.svg'.format(target_dir,name), format='svg')
        plt.tight_layout()
        plt.close()


def create_aggregated_benchmark_winrate_plot(winrate_dict, target_dir):
    for name, iterations_and_winrates in winrate_dict.items():
        iterations, winrates = zip(*iterations_and_winrates)

        iterations = iterations[0] # Checkpoint iterations are the same across experiments
        winrates = np.array(winrates)

        means, standard_deviations = winrates.mean(axis=0), winrates.std(axis=0)
        upper_bound = [mean + std for mean, std in zip(means, standard_deviations)]
        lower_bound = [mean - std for mean, std in zip(means, standard_deviations)]

        plt.errorbar(iterations, means, standard_deviations, marker='o', label=name)

        y_max = 1
        plt.xticks(iterations)
        plt.yticks(np.arange(0, y_max + y_max*0.1, y_max * 0.1)) # Watch out when changing y_max
        plt.plot((0, max(iterations)), (y_max / 2, y_max / 2), '--')

    plt.legend(loc='best')
    plt.ylabel('Average Winrate')
    plt.xlabel('Training iteration')
    plt.title('Average winrate against all opponents')
    plt.tight_layout()
    plt.savefig('{}/benchmark_winrates.svg'.format(target_dir), format='svg')
    plt.close()


def plot_single_run(run_id, source_dir):
    create_confusion_matrix_heatmaps(source_dir='{}/confusion_matrices'.format(source_dir),
                                     target_dir='{}/plots'.format(source_dir))
    plt.close()
    benchmark_winrate_dict = create_average_winrate_graph(source_dir='{}/winrates'.format(source_dir),
                                                          target_dir='{}/plots'.format(source_dir))
    plt.close()
    episodic_reward_dict = create_individual_episodic_reward_graph(source_dir='{}/episodic_rewards'.format(source_dir),
                                                                   target_dir='{}/plots'.format(source_dir))
    plt.close()
    return episodic_reward_dict, benchmark_winrate_dict


def create_confusion_matrix_heatmaps(source_dir, target_dir):
    axis_labels = [axis.replace('-', '\n') for axis in read_labels_from_file('{}/legend.txt'.format(source_dir))]
    files = all_files_in_directory(source_dir)
    for f in files:
        if f == '{}/legend.txt'.format(source_dir): continue
        create_single_heatmap(f, target_dir, axis_labels)
        plt.close()


def create_single_heatmap(source, target_dir, axis_labels):
    file_name = get_file_name_from_full_path(source)
    iteration = file_name.split('-')[-1]
    file_content = np.loadtxt(open(source, 'rb'), delimiter=', ')

    fig, ax = plt.subplots()

    v_max = 1

    ax.set_xticks(np.arange(len(axis_labels)))
    ax.set_yticks(np.arange(len(axis_labels)))
    ax.set_xticklabels(axis_labels)
    ax.set_yticklabels(axis_labels)

    image = ax.imshow(file_content, cmap='hot', vmin=0, vmax=v_max)
    fig.colorbar(image)

    # Write actual value inside of grids
    [[ax.text(j, i, file_content[i, j], ha='center', va='center', color='w')
      for j in range(len(file_content))] for i in range(len(file_content))]

    plt.title('Head to head winrates after training iteration {}'.format(iteration))
    plt.tight_layout()

    fig.savefig('{}/heatmap-{}.svg'.format(target_dir,file_name), format='svg')


def create_average_winrate_graph(source_dir, target_dir):
    benchmark_winrate_dict = {}
    for f in all_files_in_directory(source_dir):
        file_name = get_file_name_from_full_path(f)
        file_content = np.loadtxt(open(f, 'rb'), delimiter=', ')
        iterations   = file_content[:, 0]
        winrates     = file_content[:, 1]

        benchmark_winrate_dict[file_name] = (iterations, winrates)

        y_max = 1
        plt.xticks(iterations)
        plt.yticks(np.arange(0, y_max + y_max*0.1, y_max * 0.1)) # TODO refactor copied lines
        plt.plot((0, max(iterations)), (y_max / 2, y_max / 2), '--')
        plt.plot(iterations, winrates, marker='o', label=file_name)

    plt.legend(loc='best')
    plt.ylabel('Average Winrate')
    plt.xlabel('Training iteration')
    plt.title('Average winrate against all opponents')
    plt.savefig('{}/benchmark_winrates.svg'.format(target_dir), format='svg')

    return benchmark_winrate_dict


def create_individual_episodic_reward_graph(source_dir, target_dir):
    episodic_reward_dict = {}
    for f in all_files_in_directory(source_dir):
        file_name = get_file_name_from_full_path(f)
        training_scheme, algorithm = file_name.split('-')

        file_content = np.loadtxt(open(f, 'rb'), delimiter=', ')
        iterations   = file_content[:, 0]
        avg_reward   = file_content[:, 1]

        episodic_reward_dict[file_name] = avg_reward

        plt.xlabel('Training episode')
        plt.ylabel('Average episodic reward')
        plt.plot(iterations, avg_reward)

        plt.title('Average episodic reward during training\nfor policy: {}'.format(file_name))
        plt.savefig('{}/episodic_reward-{}.svg'.format(target_dir,file_name), format='svg')
        plt.close()
    return episodic_reward_dict


def all_files_in_directory(directory):
    return [join(directory, f)
            for f in os.listdir(directory) if isfile(join(directory, f))]


def get_file_name_from_full_path(filename):
    return os.path.splitext(filename)[0].split('/')[-1]


def read_labels_from_file(path):
    with open(path, 'r') as f:
        return [line.split(', ')[0] for line in f] # format: 'name, index'
