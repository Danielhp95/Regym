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
              game_buffer: Storage):
        self.num_apprentice_updates += 1

        # We are concatenating the entire datasat, this might be too memory expensive?
        s, v    = torch.cat(game_buffer.s), torch.cat(game_buffer.v)
        mcts_pi = torch.Tensor(game_buffer.normalized_child_visitations)

        # We look at number of 's' states, but we could have used anything else
        dataset_size = len(game_buffer.s)
        self.regress_against_dataset(s, v, mcts_pi, apprentice_model,
                                     indices=np.arange(dataset_size),
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
        for batch_indices in random_sample(indices, self.batch_size):
            if batches >= num_batches: break
            batches += 1

            self.optimizer.zero_grad()
            loss = self.compute_loss_from_batch(s, batch_indices, v, mcts_pi,
                                                apprentice_model)
            loss.backward()
            self.optimizer.step()

    def compute_loss_from_batch(self, s, batch_indices, v, mcts_pi, apprentice_model):
        self.num_batches_sampled += 1
        sampled_s = s[batch_indices]
        sampled_v = v[batch_indices]
        sampled_mcts_pi = mcts_pi[batch_indices]

        return compute_loss(sampled_s, sampled_mcts_pi,
                            sampled_v, apprentice_model,
                            iteration_count=self.num_batches_sampled)
