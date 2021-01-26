from typing import List, Optional, Tuple
from time import time

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
                 increase_memory_every_n_generations: int,
                 increase_memory_size_by: int,
                 end_memory_size: int,
                 use_agent_modelling: bool,
                 num_opponents: int,
                 use_cuda: bool):
        '''
        :param games_per_iteration: Number of game trajectories to be collected before training
        :param num_epochs_per_iteration: Number of times (epochs) that the
                                         entire dataset will be sampled to
                                         optimize :param: model_to_train
        :param batch_size: Number of samples to be used to compute each loss
        :param learning_rate: learning rate for the optimizer
        :param model_to_train: Model whose parameters will be updated
        :param initial_memory_size: Initial maxium memory size
        :param increase_memory_every_n_generations: Number of generations to elapse
                                               before increasing dataset size.
        :param increase_memory_size_by: Number of datapoints to increase the size
                                        of the algorithm's dataset everytime the dataset's
                                        size grows, as dictated by
                                        :param: increase_memory_every_n_generations
        :param end_memory_size: Maximum memory size
        :param use_agent_modelling: Flag to control whether to add a loss of
                                    from modelling opponent actions during training
        :param use_cuda: Wether to load tensors to an available cuda device for loss computation
        '''
        self.generation = 0
        self.games_per_iteration = games_per_iteration
        self.episodes_collected_since_last_train = 0
        self.num_batches_sampled = 0
        self.use_cuda = use_cuda

        # Init dataset
        self.memory: Storage = Storage(size=initial_memory_size)
        self.memory.add_key('normalized_child_visitations')

        self.use_agent_modelling = use_agent_modelling
        self.num_opponents = num_opponents  # TODO: maybe move this to agent?
        if self.use_agent_modelling:
            assert self.num_opponents == 1, 'Opponent modelling only supported against 1 opponent. This should have broken in ExpertIterationAgent!'
            self.memory.add_key('opponent_policy')
            self.memory.add_key('opponent_s')

        self.initial_memory_size = initial_memory_size
        self.end_memory_size = end_memory_size
        self.increase_memory_every_n_generations = increase_memory_every_n_generations
        self.increase_memory_size_by = increase_memory_size_by

        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(model_to_train.parameters(),
                                          lr=self.learning_rate)

        # To be set by an ExpertIterationAgent
        self.summary_writer: torch.util.tensorboard.SummaryWritter = None

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

    def train(self, apprentice_model: nn.Module) -> torch.Tensor:
        ''' Highest level function
        :param apprentice_model: Model to train
        :returns: Total loss computed
        '''
        start_time = time()
        self.generation += 1
        self.episodes_collected_since_last_train = 0

        self.update_storage(self.memory, self.memory.size)

        s, v, mcts_pi, opponent_s, opponent_policy = self.preprocess_memory(
            self.memory)

        # We look at number of 's' states, but we could have used anything else
        dataset_size = len(self.memory.s)
        generation_loss = self.regress_against_dataset(
            s,
            v,
            mcts_pi,
            opponent_policy,
            opponent_s,
            apprentice_model,
            indices=np.arange(dataset_size),
            batch_size=self.batch_size,
            num_epochs=self.num_epochs_per_iteration)

        if self.summary_writer:
            self.summary_writer.add_scalar('Timing/Generation_update', time() - start_time,
                                           self.generation)
            self.summary_writer.add_scalar('Training/Memory_size', dataset_size,
                                           self.generation)
            self.summary_writer.add_scalar('Training/Total_generation_loss',
                                           generation_loss.cpu(),
                                           self.generation)
        return generation_loss

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
                                num_epochs: int) -> torch.Tensor:
        '''
        Updates :param apprentice_model: netowrk parameters to better predict:
            - State value function: :param: s, :param: v
            - Expert policy: :param: s, :param: mcts_pi
        Samples :param: num_batches of size :param: batch_size from list of
        :param: indices.
        :returns: Total (Aggregated) loss computed over :param: num_epochs.
        '''
        # Sneaky-hacky way of getting device
        inititial_device = next(apprentice_model.parameters())
        apprentice_model.to('cuda' if self.use_cuda else 'cpu')
        total_loss = 0.
        for e in range(num_epochs):
            for batch_indices in random_sample(indices, batch_size):
                self.num_batches_sampled += 1

                if self.use_agent_modelling:
                    opponent_policy_batch = opponent_policy[batch_indices].to('cuda' if self.use_cuda else 'cpu')
                    opponent_s_batch = opponent_s[batch_indices].to('cuda' if self.use_cuda else 'cpu')
                else:
                    opponent_policy_batch, opponent_s_batch = None, None

                loss = compute_loss(s[batch_indices].to('cuda' if self.use_cuda else 'cpu'),
                                    mcts_pi[batch_indices].to('cuda' if self.use_cuda else 'cpu'),
                                    v[batch_indices].to('cuda' if self.use_cuda else 'cpu'),
                                    opponent_policy=opponent_policy_batch,
                                    opponent_s=opponent_s_batch,
                                    use_agent_modelling=self.use_agent_modelling,
                                    apprentice_model=apprentice_model,
                                    iteration_count=self.num_batches_sampled,
                                    summary_writer=self.summary_writer)

                # Name a more iconic trio
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.cpu().detach()
        apprentice_model.to(inititial_device)
        return total_loss

    def update_storage(self, dataset, max_memory):
        self.update_storage_size(dataset)
        # TODO: Remove duplicates when implemented correctly
        self.curate_dataset(dataset, dataset.size)

    def remove_duplicates_from_dataset(self, dataset: Storage) -> int:
        '''
        TODO: no longer using this function

        As remove_duplicates for, say,
        key 's' would also remove the value for all keys at the indices where
        's' was duplicated. This was OK previously when only cared for
        's', 'normalized_child_visitations' and 'V', but this is unacceptable
        when we also have independent keys like 'opponent_s' and 'opponent_policy'.
        '''
        exit_keys_to_average_over = ['normalized_child_visitations', 'V']
        opponent_modelling_keys_to_average_over = ['opponent_policy']
        duplicates = 0

        # Will there be an issue if we try to average over 'opponent_policy' when there are NaNs?
        # This remove_duplicates call is destroying valuable  datapoints
        duplicates += dataset.remove_duplicates(
            target_key='s',
            avg_keys=exit_keys_to_average_over
        )
        if self.use_agent_modelling:
            duplicates += dataset.remove_duplicates(
                target_key='opponent_s',
                avg_keys=opponent_modelling_keys_to_average_over
            )
        return duplicates

    def update_storage_size(self, dataset):
        ''' Increases maximum size of dataset if required '''
        if self.generation % self.increase_memory_every_n_generations == 0 \
                and dataset.size < self.end_memory_size:
            dataset.size += self.increase_memory_size_by

    def curate_dataset(self, dataset: Storage, max_memory: int):
        '''
        Removes old experiences from :param: dataset so that it keeps at most
        :param: max_memory datapoints in it.

        ASSUMPTION: ALL "keys" have the same number of datapoints
        '''
        oversize = max(len(dataset.s) - dataset.size, 0)
        if oversize > 0:
            for k in dataset.non_empty_keys(): del getattr(dataset, k)[:oversize]
        assert len(dataset.s) <= max_memory
        if self.summary_writer: self.summary_writer.add_scalar(
            'Training/Memory_oversize_at_generation', oversize, self.generation)

    def __getstate__(self):
        '''
        Function invoked when pickling.

        torch.utils.SummaryWriters are not pickable.
        '''
        to_pickle_dict = self.__dict__
        if 'summary_writer' in to_pickle_dict:
            to_pickle_dict = self.__dict__.copy()
            to_pickle_dict['summary_writer'] = None
        return to_pickle_dict

    def __repr__(self):
        gen_stats = f'Generation: {self.generation}\nGames per generation: {self.games_per_iteration}\nEpisodes since last generation: {self.episodes_collected_since_last_train}\n'
        train_stats = f'Batches sampled: {self.num_batches_sampled}\nBatch size: {self.batch_size}\nLearning rate: {self.learning_rate}\nEpochs per generation: {self.num_epochs_per_iteration}\n'
        memory_stats = f'Initial memory size: {self.initial_memory_size}\nMemory increase frequency: {self.increase_memory_every_n_generations}\nMax memory size: {self.end_memory_size}\n'
        return gen_stats + train_stats + memory_stats + f'{self.memory}\n' + f'Use CUDA: {self.use_cuda}'
