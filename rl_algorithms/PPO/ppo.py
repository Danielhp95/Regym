from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..replay_buffers import Storage
from ..networks import random_sample


class PPOAlgorithm():

    def __init__(self, kwargs):
        '''
        horizon:
        discount:
        use_gae:
        use_cuda:
        gae_tau:
        entropy_weight:
        gradient_clip:
        optimization_epochs:
        mini_batch_size:
        ppo_ratio_clip:
        learning_rate:
        adam_eps:
        model:
        replay_buffer:
        state_preprocess:
        "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
        "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
        "use_cuda": boolean to specify whether to use CUDA.
        '''
        self.kwargs = deepcopy(kwargs)
        self.model = self.kwargs['model']
        if self.kwargs['use_cuda']:
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['learning_rate'], eps=kwargs['adam_eps'])
        self.storage = Storage(self.kwargs['horizon'])

    def train(self):
        '''
        2. Calculates values to regress towards
        '''
        last_visited_state = self.storage.s[-1]
        prediction = self.model(last_visited_state)
        self.storage.add(prediction)
        self.storage.placeholder()

        advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32)) # TODO explain (used to be number of workers)
        returns = prediction['v'].detach()
        for i in reversed(range(self.kwargs['horizon'])):
            returns = self.storage.r[i] + self.kwargs['discount'] * self.storage.non_terminal[i] * returns
            if not self.kwargs['use_gae']:
                advantages = returns - self.storage.v[i].detach()
            else:
                td_error = self.storage.r[i] + self.kwargs['discount'] * self.storage.non_terminal[i] * self.storage.v[i + 1] - self.storage.v[i]
                advantages = advantages * self.kwargs['use_gae'] * self.kwargs['discount'] * self.storage.non_terminal[i] + td_error
            self.storage.adv[i] = advantages.detach()
            self.storage.ret[i] = returns.detach()

        states, actions, log_probs_old, returns, advantages = self.storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(self.kwargs['optimization_epochs']):
            self.optimize_model(states, actions, log_probs_old, returns, advantages)

        self.storage.reset()

    def optimize_model(self, states, actions, log_probs_old, returns, advantages):
        sampler = random_sample(np.arange(states.size(0)), self.kwargs['mini_batch_size'])
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_log_probs_old = log_probs_old[batch_indices].cuda() if self.kwargs['use_cuda'] else log_probs_old[batch_indices]
            sampled_returns = returns[batch_indices].cuda() if self.kwargs['use_cuda'] else returns[batch_indices]
            sampled_advantages = advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else advantages[batch_indices]

            # TODO make sure prediction is cuda
            prediction = self.model(sampled_states, sampled_actions)
            ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
            obj = ratio * sampled_advantages
            obj_clipped = ratio.clamp(1.0 - self.kwargs['ppo_ratio_clip'],
                                      1.0 + self.kwargs['ppo_ratio_clip']) * sampled_advantages
            policy_loss = -torch.min(obj, obj_clipped).mean() - self.kwargs['entropy_weight'] * prediction['ent'].mean()

            value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
            self.optimizer.step()
