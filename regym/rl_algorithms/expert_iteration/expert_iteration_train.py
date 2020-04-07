from typing import List

import numpy as np

import torch
import torch.nn as nn

from regym.rl_algorithms.networks import random_sample
from regym.rl_algorithms.replay_buffers import Storage

from regym.rl_algorithms.expert_iteration.expert_iteration_loss import compute_loss


class ExpertIterationAlgorithm():

    def __init__(self, batch_size, batches_per_train_iteration: int,
                 learning_rate: float,
                 model_to_train: nn.Module):
        '''

        :param batches_per_train_iteration: Number of mini batches to sample
                                            for every training iteration
        :param batch_size: TODO
        :param model_to_train: TODO say that we only want it to get its parameters
        :param learning_rate: learning rate for the optimizer
        '''
        self.batches_per_train_iteration = batches_per_train_iteration
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(model_to_train.parameters(),
                                          lr=self.learning_rate)

        self.num_apprentice_updates = 0
        self.num_batches_sampled = 0

    def train(self, apprentice_model: nn.Module,
              dataset: Storage):
        '''
        Highest level function.
        '''
        self.num_apprentice_updates += 1

        self.curate_dataset(dataset, dataset.size,
                            keys=['s', 'v', 'normalized_child_visitations'])

        # We are concatenating the entire datasat, this might be too memory expensive?
        s, v    = torch.cat(dataset.s), torch.cat(dataset.v)
        mcts_pi = torch.Tensor(dataset.normalized_child_visitations)

        # We look at number of 's' states, but we could have used anything else
        dataset_size = len(dataset.s)
        self.regress_against_dataset(s, v, mcts_pi, apprentice_model,
                                     indices=np.arange(dataset_size),
                                     batch_size=self.batch_size,
                                     num_batches=self.batches_per_train_iteration)

    def regress_against_dataset(self, s, v, mcts_pi, apprentice_model,
                                indices: List[int], batch_size: int,
                                num_batches: int):
        '''
        Updates :param apprentice_model: netowrk parameters to better predict:
            - State value function: :param: s, :param: v
            - Expert policy: :param: s, :param: mcts_pi
        Samples :param: num_batches of size :param: batch_size from list of
        :param: indices.
        '''
        batches = 0
        for batch_indices in random_sample(indices, batch_size):
            if batches >= num_batches: break
            self.num_batches_sampled += 1
            batches += 1

            self.optimizer.zero_grad()
            loss = compute_loss(s[batch_indices],
                                mcts_pi[batch_indices],
                                v[batch_indices],
                                apprentice_model,
                                iteration_count=self.num_batches_sampled)
            loss.backward()
            self.optimizer.step()

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
