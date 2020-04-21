from typing import List

import numpy as np

import torch
import torch.nn as nn

from regym.rl_algorithms.networks import random_sample
from regym.rl_algorithms.replay_buffers import Storage

from regym.rl_algorithms.expert_iteration.expert_iteration_loss import compute_loss


class ExpertIterationAlgorithm():

    def __init__(self, batch_size, num_epochs_per_iteration: int,
                 learning_rate: float,
                 games_per_iteration: int,
                 initial_memory_size: int,
                 memory_size_increase_frequency: int,
                 end_memory_size: int,
                 model_to_train: nn.Module):
        '''
        :param batch_size: TODO
        :param model_to_train: TODO say that we only want it to get its parameters
        :param learning_rate: learning rate for the optimizer
        :param games_per_iteration: Number of game trajectories to be collected before training
        :param initial_memory_size: TODO
        :param memory_size_increase_frequency: (Int) Number of iterations to elapse before increasing dataset size.
        :param end_memory_size: TODO
        '''
        self.games_per_iteration = games_per_iteration

        # Init dataset
        self.memory = Storage(size=initial_memory_size)
        self.memory.add_key('normalized_child_visitations')

        self.initial_memory_size = initial_memory_size
        self.memory_size_increase_frequency = memory_size_increase_frequency
        self.end_memory_size = end_memory_size

        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(model_to_train.parameters(),
                                          lr=self.learning_rate)

        self.episodes_collected_since_last_train = 0
        self.generation = 0
        self.num_batches_sampled = 0

    def should_train(self):
        return self.episodes_collected_since_last_train >= self.games_per_iteration

    def add_episode_trajectory(self, episode_trajectory: Storage):
        self.episodes_collected_since_last_train += 1
        self.memory.add({'normalized_child_visitations': episode_trajectory.normalized_child_visitations,
                         's': episode_trajectory.s,
                         'v': episode_trajectory.v})

    def train(self, apprentice_model: nn.Module):
        ''' Highest level function '''
        self.episodes_collected_since_last_train = 0
        self.generation += 1

        # TODO: later make this function:
        # - Remove duplicates
        self.update_storage(self.memory, self.memory.size,
                            keys=['s', 'v', 'normalized_child_visitations'])

        # We are concatenating the entire datasat, this might be too memory expensive?
        s, v    = torch.cat(self.memory.s), torch.cat(self.memory.v)
        mcts_pi = torch.stack(self.memory.normalized_child_visitations)

        # We look at number of 's' states, but we could have used anything else
        dataset_size = len(self.memory.s)
        self.regress_against_dataset(s, v, mcts_pi, apprentice_model,
                                     indices=np.arange(dataset_size),
                                     batch_size=self.batch_size,
                                     num_epochs=self.num_epochs_per_iteration)

    def regress_against_dataset(self, s, v, mcts_pi, apprentice_model,
                                indices: List[int], batch_size: int,
                                num_epochs: int):
        '''
        Updates :param apprentice_model: netowrk parameters to better predict:
            - State value function: :param: s, :param: v
            - Expert policy: :param: s, :param: mcts_pi
        Samples :param: num_batches of size :param: batch_size from list of
        :param: indices.
        '''
        for e in range(num_epochs):
            for batch_indices in random_sample(indices, batch_size):
                self.num_batches_sampled += 1

                loss = compute_loss(s[batch_indices],
                                    mcts_pi[batch_indices],
                                    v[batch_indices],
                                    apprentice_model,
                                    iteration_count=self.num_batches_sampled)

                # Name a more iconic trio
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def update_storage(self, dataset, max_memory, keys):
        self.update_storage_size(dataset)
        self.curate_dataset(dataset, dataset.size,
                            keys=['s', 'v', 'normalized_child_visitations'])

    def update_storage_size(self, dataset):
        '''
        Increases maximum size of dataset if required
        '''
        if self.generation % self.memory_size_increase_frequency == 0 \
                and dataset.size < self.end_memory_size:
            dataset.size += self.initial_memory_size

    def curate_dataset(self, dataset: Storage, max_memory: int,
                       keys: List[str]):
        '''
        Removes old experiences from :param: dataset so that it keeps at most
        :param: max_memory datapoints in it.

        ASSUMPTION: ALL "keys" have the same number of datapoints
        '''
        oversize = len(dataset.s) - dataset.size
        if oversize > 0:
            for k in keys: del getattr(dataset, k)[:oversize]
        assert len(dataset.s) <= max_memory
