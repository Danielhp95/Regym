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
                 initial_memory_size: int,
                 memory_size_increase_frequency: int,
                 end_memory_size: int,
                 model_to_train: nn.Module):
        '''
        :param batch_size: TODO
        :param model_to_train: TODO say that we only want it to get its parameters
        :param learning_rate: learning rate for the optimizer

        :param initial_memory_size: TODO
        :param memory_size_increase_frequency: (Int) Number of iterations to elapse before increasing dataset size.
        :param end_memory_size: TODO
        '''
        self.initial_memory_size = initial_memory_size
        self.memory_size_increase_frequency = memory_size_increase_frequency
        self.end_memory_size = end_memory_size

        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(model_to_train.parameters(),
                                          lr=self.learning_rate)

        self.generation = 0
        self.num_batches_sampled = 0

    def train(self, apprentice_model: nn.Module,
              dataset: Storage):
        '''
        Highest level function.
        '''
        self.generation += 1

        # TODO: later make this function:
        # - Enlarge dataset up to a max
        # - Remove duplicates
        self.update_storage(dataset, dataset.size,
                            keys=['s', 'v', 'normalized_child_visitations'])

        # We are concatenating the entire datasat, this might be too memory expensive?
        s, v    = torch.cat(dataset.s), torch.cat(dataset.v)
        mcts_pi = torch.Tensor(dataset.normalized_child_visitations)

        # We look at number of 's' states, but we could have used anything else
        dataset_size = len(dataset.s)
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
