from functools import reduce
from typing import List, Tuple
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from regym.rl_algorithms.replay_buffers import Storage
from regym.networks import random_sample

from regym.rl_algorithms.PPO.ppo_loss import compute_loss


class PPOAlgorithm():

    def __init__(self, kwargs, model):
        '''
        Refer to original paper for further explanation: https://arxiv.org/pdf/1707.06347.pdf
        horizon: (0, infinity) Number of timesteps that will elapse in between optimization calls.
        discount: (0,1) Reward discount factor
        use_gae: Flag, wether to use Generalized Advantage Estimation (GAE) (instead of return base estimation)
        gae_tau: (0,1) GAE hyperparameter.
        use_cuda: Flag, to specify whether to use CUDA tensors in Pytorch calculations
        entropy_weight: (0,1) Coefficient for (regularatization) entropy based loss
        gradient_clip: float, Clips gradients to reduce the chance of destructive updates
        optimization_epochs: int, Number of epochs per optimization step.
        mini_batch_size: int, Mini batch size to use to calculate losses (Use power of 2 for efficciency)
        ppo_ratio_clip: float, clip boundaries (1 - clip, 1 + clip) used in clipping loss function.
        learning_rate: float, optimizer learning rate.
        adam_eps: (float), Small Epsilon value used for ADAM optimizer. Prevents numerical instability when v^{hat} (Second momentum estimator) is near 0.
        model: (Pytorch nn.Module) Used to represent BOTH policy network and value network
        '''
        self.kwargs = deepcopy(kwargs)
        self.horizon = kwargs['horizon']
        self.discount = kwargs['discount']
        self.use_gae = kwargs['use_gae']
        self.gae_tau = kwargs['gae_tau']
        self.ppo_ratio_clip = kwargs['ppo_ratio_clip']
        self.entropy_weight = kwargs['entropy_weight']
        self.optimization_epochs = kwargs['optimization_epochs']
        self.learning_rate = kwargs['learning_rate']
        self.gradient_clip = kwargs['gradient_clip']
        self.adam_eps = kwargs['adam_eps']
        self.use_cuda = kwargs['use_cuda']
        self.batch_size = kwargs['mini_batch_size']
        self.model = model
        if self.use_cuda:
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=self.adam_eps)

        self.recurrent = False
        self.rnn_keys = [key for key,value in self.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.rnn_keys):
            self.recurrent = True

        # As a default, create a single storage
        self.storages: List[Storage] = self.create_storages(num_storages=1)

        # Number of times train was called
        self.num_updates = 0

        # Number of policy updates (Number of times compute_loss() was called)
        self.num_optimizer_steps = 0

    def create_storages(self, num_storages: int, size=-1) -> List[Storage]:
        if size == -1: size = self.horizon
        storages = [Storage(size) for _ in range(num_storages)]
        if self.recurrent:
            for storage in storages:
                storage.add_key('rnn_states')
                storage.add_key('next_rnn_states')
        return storages

    def train(self):
        self.num_updates += 1
        for storage in self.storages:
            storage.placeholder(num_elements=len(storage.s))
            self.compute_advantages_and_returns(storage)

        (states, actions, log_probs_old, returns,
         advantages, rnn_states) = self.retrieve_values_from_storages()

        if self.recurrent: rnn_states = self.reformat_rnn_states(rnn_states)

        for _ in range(self.optimization_epochs):
            self.optimize_model(states, actions, log_probs_old, returns,
                                advantages, rnn_states)

        for storage in self.storages: storage.reset()

    def reformat_rnn_states(self, rnn_states):
        reformated_rnn_states = { k: ( [list()], [list()] ) for k in self.rnn_keys }
        for rnn_state in rnn_states:
            for k, (hstates, cstates) in rnn_state.items():
                for idx_layer, (h,c) in enumerate(zip(hstates, cstates)):
                    reformated_rnn_states[k][0][0].append(h)
                    reformated_rnn_states[k][1][0].append(c)
        for k, (hstates, cstates) in reformated_rnn_states.items():
            hstates = torch.cat(hstates[0], dim=0)
            cstates = torch.cat(cstates[0], dim=0)
            reformated_rnn_states[k] = ([hstates], [cstates])
        return reformated_rnn_states

    def compute_advantages_and_returns(self, storage):
        advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32)) # TODO explain (used to be number of workers)
        returns = storage.V[-1].detach()
        num_observed_rewards = len(storage.r)
        for i in reversed(range(num_observed_rewards)):
            returns = storage.r[i] + self.discount * storage.non_terminal[i] * returns
            if not self.use_gae:
                advantages = returns - storage.V[i].detach()
            else:
                td_error = storage.r[i] + self.discount * storage.non_terminal[i] * storage.V[i + 1] - storage.V[i]
                advantages = advantages * self.gae_tau * self.discount * storage.non_terminal[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

    def retrieve_values_from_storages(self):
        all_states = []
        all_actions = []
        all_log_probs_old = []
        all_returns = []
        all_advantages = []
        all_rnn_states = [] if self.recurrent else None

        for storage in self.storages:
            (states, actions, log_probs_old, returns, advantages,
             rnn_states) = self.retrieve_values_from_single_storage(storage)
            all_states += states
            all_actions += actions
            all_log_probs_old += log_probs_old
            all_returns += returns
            all_advantages += advantages
            if self.recurrent: all_rnn_states += rnn_states

        all_states = torch.stack(all_states, dim=0)
        all_actions = torch.stack(all_actions, dim=0)
        all_log_probs_old = torch.stack(all_log_probs_old, dim=0)
        all_returns = torch.stack(all_returns, dim=0)
        all_advantages = torch.stack(all_advantages, dim=0)
        if self.recurrent: all_rnn_states = torch.stack(all_rnn_states, dim=0)
        return (all_states, all_actions, all_log_probs_old, all_returns,
               all_advantages, all_rnn_states)

    def retrieve_values_from_single_storage(self, storage):
        if self.recurrent:
            cat = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv', 'rnn_states'])
            rnn_states = cat[-1]
            states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=1), cat[:-1])
        else:
            states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0),
                                                                     storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv']) )
            rnn_states = None

        # Turning actions into dim: self.horizon x 1
        # (Might break in environments where actions are not a single number)
        assert storage.s != self.horizon, (f'Number of states in storage ({len(storage.s)}) '
                                           f"should be equal to PPO's horizon ({self.horizon})")
        #states = states.view(len(storage.s), -1)
        actions = actions.view(-1, 1)
        log_probs_old = log_probs_old.view(-1, 1)
        returns = returns.view(-1, 1)
        advantages = advantages.view(-1, 1)
        advantages = self.standardize(advantages)
        return states, actions, log_probs_old, returns, advantages, rnn_states

    def standardize(self, x):
        return (x - x.mean()) / x.std()

    def sample_batch_from_indices(self, batch_indices, states, actions,
                                  log_probs_old, returns, advantages,
                                  rnn_states) -> Tuple[torch.Tensor]:
        sampled_states = states[batch_indices].cuda() if self.use_cuda else states[batch_indices]
        sampled_actions = actions[batch_indices].cuda() if self.use_cuda else actions[batch_indices]
        sampled_log_probs_old = log_probs_old[batch_indices].cuda() if self.use_cuda else log_probs_old[batch_indices]
        sampled_returns = returns[batch_indices].cuda() if self.use_cuda else returns[batch_indices]
        sampled_advantages = advantages[batch_indices].cuda() if self.use_cuda else advantages[batch_indices]

        sampled_rnn_states = None
        if self.recurrent:
            sampled_rnn_states = { k: ([None]*nbr_layers_per_rnn[k] , [None]*nbr_layers_per_rnn[k]) for k in self.rnn_keys}
            for k in sampled_rnn_states:
                for idx in range(nbr_layers_per_rnn[k]):
                    sampled_rnn_states[k][0][idx] = rnn_states[k][0][idx][batch_indices].cuda() if self.use_cuda else rnn_states[k][0][idx][batch_indices]
                    sampled_rnn_states[k][1][idx] = rnn_states[k][1][idx][batch_indices].cuda() if self.use_cuda else rnn_states[k][1][idx][batch_indices]
        return (sampled_states, sampled_actions, sampled_log_probs_old,
                sampled_returns, sampled_advantages, sampled_rnn_states)

    def optimize_model(self, states, actions, log_probs_old, returns, advantages, rnn_states=None):
        sampler = random_sample(np.arange(states.size(0)), self.batch_size)
        assert( (self.recurrent and rnn_states is not None) or not(self.recurrent or rnn_states is not None) )
        if self.recurrent:
            nbr_layers_per_rnn = { k:len(rnn_states[k][0] ) for k in self.rnn_keys}

        for batch_indices in sampler:
            self.num_optimizer_steps += 1
            batch_indices = torch.from_numpy(batch_indices).long()

            (sampled_states, sampled_actions, sampled_log_probs_old,
            sampled_returns, sampled_advantages,
            sampled_rnn_states) = self.sample_batch_from_indices(batch_indices,
                    states, actions, log_probs_old, returns, advantages,
                    rnn_states)

            total_loss = compute_loss(states=sampled_states,
                                      actions=sampled_actions,
                                      log_probs_old=sampled_log_probs_old,
                                      returns=sampled_returns,
                                      advantages=sampled_advantages,
                                      model=self.model,
                                      rnn_states=sampled_rnn_states,
                                      ratio_clip=self.ppo_ratio_clip,
                                      entropy_weight=self.entropy_weight,
                                      iteration_count=self.num_optimizer_steps)

            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()


    def __repr__(self):
        return (f'Num updates: {self.num_updates}\n'
         f'Num optimizer steps: {self.num_optimizer_steps}\n'
         f'Horizon: {self.horizon}\n'
         f'Discount: {self.discount}\n'
         f'Use_gae: {self.use_gae}\n'
         f'Gae tau: {self.gae_tau}\n'
         f'Ppo ratio clip: {self.ppo_ratio_clip}\n'
         f'Entropy weight: {self.entropy_weight}\n'
         f'Optimization epochs: {self.optimization_epochs}\n'
         f'Learning rate: {self.learning_rate}\n'
         f'Gradient clip: {self.gradient_clip}\n'
         f'Adam eps: {self.adam_eps}\n'
         f'Batch size: {self.batch_size}\n'
         f'Use cuda: {self.use_cuda}\n'
         f'Storages: {self.storages}'
         )
