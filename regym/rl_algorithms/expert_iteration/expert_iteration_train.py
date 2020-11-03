from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn

from regym.networks import random_sample
from regym.rl_algorithms.replay_buffers import Storage

from regym.rl_algorithms.expert_iteration.expert_iteration_loss import compute_loss


class ExpertIterationAlgorithm():

    def __init__(self, games_per_iteration: int,
                 num_epochs_per_iteration: int,
                 batch_size: int,
                 learning_rate: float,
                 model_to_train: nn.Module,
                 initial_memory_size: int,
                 memory_size_increase_frequency: int,
                 end_memory_size: int,
                 use_agent_modelling: bool,
                 num_opponents: int):
        '''
        :param games_per_iteration: Number of game trajectories to be collected before training
        :param num_epochs_per_iteration: Number of times (epochs) that the
                                         entire dataset will be sampled to
                                         optimize :param: model_to_train
        :param batch_size: Number of samples to be used to compute each loss
        :param learning_rate: learning rate for the optimizer
        :param model_to_train: Model whose parameters will be updated
        :param initial_memory_size: Initial maxium memory size
        :param memory_size_increase_frequency: Number of generations to elapse
                                               before increasing dataset size.
        :param end_memory_size: Maximum memory size
        :param use_agent_modelling: Flag to control whether to add a loss of
                                    from modelling opponent actions during training
        '''
        self.generation = 0
        self.games_per_iteration = games_per_iteration
        self.episodes_collected_since_last_train = 0
        self.num_batches_sampled = 0

        # Init dataset
        self.memory = Storage(size=initial_memory_size)
        self.memory.add_key('normalized_child_visitations')

        self.use_agent_modelling = use_agent_modelling
        self.num_opponents = num_opponents  # TODO: maybe move this to agent?
        if self.use_agent_modelling:
            assert self.num_opponents == 1, 'Opponent modelling only supported against 1 opponent. This should have broken in ExpertIterationAgent!'
            self.memory.add_key('opponent_policy')
            self.memory.add_key('opponent_s')

        self.initial_memory_size = initial_memory_size
        self.memory_size_increase_frequency = memory_size_increase_frequency
        self.end_memory_size = end_memory_size

        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(model_to_train.parameters(),
                                          lr=self.learning_rate)

    def should_train(self):
        return self.episodes_collected_since_last_train >= self.games_per_iteration

    def add_episode_trajectory(self, episode_trajectory: Storage):
        self.episodes_collected_since_last_train += 1
        dataset_size = len(episode_trajectory.s)
        for i in range(dataset_size):
            self.memory.add({'normalized_child_visitations': episode_trajectory.normalized_child_visitations[i],
                             's': episode_trajectory.s[i],
                             'V': episode_trajectory.V[i]})
            if self.use_agent_modelling:
                self.memory.add({'opponent_policy': episode_trajectory.opponent_policy[i]})
                self.memory.add({'opponent_s': episode_trajectory.opponent_s[i]})

    def train(self, apprentice_model: nn.Module):
        ''' Highest level function '''
        self.episodes_collected_since_last_train = 0
        self.generation += 1

        self.update_storage(self.memory, self.memory.size,
                            keys=['s', 'V', 'normalized_child_visitations'])

        s, v, mcts_pi, opponent_s, opponent_policy = self.preprocess_memory(
            self.memory)

        # We look at number of 's' states, but we could have used anything else
        dataset_size = len(self.memory.s)
        self.regress_against_dataset(
            s,
            v,
            mcts_pi,
            opponent_policy,
            opponent_s,
            apprentice_model,
            indices=np.arange(dataset_size),
            batch_size=self.batch_size,
            num_epochs=self.num_epochs_per_iteration)

    def preprocess_memory(self, memory: Storage) -> Tuple:
        # We are concatenating the entire datasat, this might be too memory expensive?
        s, v    = torch.cat(self.memory.s), torch.cat(self.memory.V)
        mcts_pi = torch.stack(self.memory.normalized_child_visitations)
        if self.use_agent_modelling:
            opponent_s = torch.cat(self.memory.opponent_s)
            opponent_policy = torch.stack(self.memory.opponent_policy)
        else:
            opponent_policy, opponent_s = None, None
        return s, v, mcts_pi, opponent_s, opponent_policy

    def regress_against_dataset(self, s: torch.FloatTensor,
                                v: torch.FloatTensor,
                                mcts_pi: torch.FloatTensor,
                                opponent_policy: Optional[torch.FloatTensor],
                                opponent_s: Optional[torch.FloatTensor],
                                apprentice_model: nn.Module,
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

                if self.use_agent_modelling:
                    opponent_policy_batch = opponent_policy[batch_indices]
                    opponent_s_batch = opponent_s[batch_indices]
                else:
                    opponent_policy_batch, opponent_s_batch = None, None

                loss = compute_loss(s[batch_indices],
                                    mcts_pi[batch_indices],
                                    v[batch_indices],
                                    opponent_policy=opponent_policy_batch,
                                    opponent_s=opponent_s_batch,
                                    use_agent_modelling=self.use_agent_modelling,
                                    apprentice_model=apprentice_model,
                                    iteration_count=self.num_batches_sampled)

                # Name a more iconic trio
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def update_storage(self, dataset, max_memory, keys):
        self.update_storage_size(dataset)
        avg_keys = ['normalized_child_visitations', 'V'] + (['opponent_policy'] if self.use_agent_modelling else [])
        # Will there be an issue if we try to average over 'opponent_policy' when there are NaNs?
        dataset.remove_duplicates(target_key='s',
                                  avg_keys=avg_keys)
        self.curate_dataset(dataset, dataset.size,
                            keys=['s', 'V', 'normalized_child_visitations',
                                  'opponent_policy', 'opponent_s'])

    def update_storage_size(self, dataset):
        ''' Increases maximum size of dataset if required '''
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

    def __repr__(self):
        gen_stats = f'Generation: {self.generation}\nGames per generation: {self.games_per_iteration}\nEpisodes since last generation: {self.episodes_collected_since_last_train}\n'
        train_stats = f'Batches sampled: {self.num_batches_sampled}\nBatch size: {self.batch_size}\nLearning rate: {self.learning_rate}\nEpochs per generation: {self.episodes_collected_since_last_train}\n'
        memory_stats = f'Initial memory size: {self.initial_memory_size}\nMemory increase frequency: {self.memory_size_increase_frequency}\nMax memory size: {self.end_memory_size}\n'
        return gen_stats + train_stats + memory_stats + str(self.memory)

