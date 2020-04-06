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

    def train(self, apprentice_model: nn.Module,
              game_buffer: Storage):
        # We look at number of 's' states, but we could have used anything else.

        # We are concatenating the entire datasat, this might be too memory expensive?
        s, v    = torch.cat(game_buffer.s), torch.cat(game_buffer.v)
        mcts_pi = torch.Tensor(game_buffer.normalized_child_visitations)

        dataset_size = len(game_buffer.s)
        for i in range(self.batches_per_train_iteration):
            # Prepare batch
            batch = np.random.randint(dataset_size, size=self.batch_size)
            sampled_s = s[batch]
            sampled_v = v[batch]
            sampled_mcts_pi = mcts_pi[batch]

            # Compute loss from tensored batch
            self.optimizer.zero_grad()
            loss = compute_loss(sampled_s, sampled_mcts_pi,
                                sampled_v, apprentice_model)
