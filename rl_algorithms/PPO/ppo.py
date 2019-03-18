from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..replay_buffers import Storage
from ..networks import random_sample


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
        adam_eps: (float), Small Epsilon value used for ADAM optimizar. Prevents numerical inestability when v^{hat} (Second momentum estimator) is near 0.
        model: (Pytorch nn.Module) Used to represent BOTH policy network and value network
        '''
        self.kwargs = deepcopy(kwargs)
        self.model = model
        if self.kwargs['use_cuda']:
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['learning_rate'], eps=kwargs['adam_eps'])
        self.storage_capacity = self.kwargs['horizon']
        if 'nbr_actor' in self.kwargs: self.storage_capacity *= self.kwargs['nbr_actor']
        self.storage = Storage(self.storage_capacity)

    def train(self):
        self.storage.placeholder()

        self.compute_advantages_and_returns()
        states, actions, log_probs_old, returns, advantages = self.retrieve_values_from_storage()
        #progress_bar = tqdm(range(self.kwargs['optimization_epochs']) )
        #for epoch in progress_bar:
        for epoch in range(self.kwargs['optimization_epochs']):
            self.optimize_model(states, actions, log_probs_old, returns, advantages)
            #progress_bar.set_description(f"Training epoch : {epoch}/{self.kwargs['optimization_epochs']}")
    
        self.storage.reset()

    def compute_advantages_and_returns(self):
        advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32)) # TODO explain (used to be number of workers)
        returns = self.storage.v[-1].detach()
        for i in reversed(range(self.storage_capacity)):
            returns = self.storage.r[i] + self.kwargs['discount'] * self.storage.non_terminal[i] * returns
            if not self.kwargs['use_gae']:
                advantages = returns - self.storage.v[i].detach()
            else:
                td_error = self.storage.r[i] + self.kwargs['discount'] * self.storage.non_terminal[i] * self.storage.v[i + 1] - self.storage.v[i]
                advantages = advantages * self.kwargs['gae_tau'] * self.kwargs['discount'] * self.storage.non_terminal[i] + td_error
            self.storage.adv[i] = advantages.detach()
            self.storage.ret[i] = returns.detach()

    def retrieve_values_from_storage(self):
        states, actions, log_probs_old, returns, advantages = self.storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
        advantages = self.standardize(advantages)
        return states, actions, log_probs_old, returns, advantages

    def standardize(self, x):
        return (x - x.mean()) / x.std()

    def optimize_model(self, states, actions, log_probs_old, returns, advantages):
        sampler = random_sample(np.arange(states.size(0)), self.kwargs['mini_batch_size'])
        #nbr_batch = states.size(0)//self.kwargs['mini_batch_size']
        #progress_bar = tqdm(range(nbr_batch) )
        #for it,batch_indices in zip(progress_bar,sampler):
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_log_probs_old = log_probs_old[batch_indices].cuda() if self.kwargs['use_cuda'] else log_probs_old[batch_indices]
            sampled_returns = returns[batch_indices].cuda() if self.kwargs['use_cuda'] else returns[batch_indices]
            sampled_advantages = advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else advantages[batch_indices]

            prediction = {k:v.view((self.kwargs['mini_batch_size'],-1)) for k,v in self.model(sampled_states, sampled_actions).items() }
            #prediction = self.model(sampled_states, sampled_actions)

            ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
            obj = ratio * sampled_advantages
            obj_clipped = ratio.clamp(1.0 - self.kwargs['ppo_ratio_clip'],
                                      1.0 + self.kwargs['ppo_ratio_clip']) * sampled_advantages
            policy_loss = -torch.min(obj, obj_clipped).mean() - self.kwargs['entropy_weight'] * prediction['ent'].mean() # L^{clip} and L^{S} from original paper

            value_loss = 0.5 * (sampled_returns - prediction['v']).pow(2).mean()

            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
            self.optimizer.step()
            #progress_bar.set_description(f"Epoch: training iteration : {it}/{nbr_batch}")

