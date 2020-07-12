from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from regym.rl_algorithms.replay_buffers import Storage
from regym.networks import random_sample


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
        self.model = model
        if self.kwargs['use_cuda']:
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['learning_rate'], eps=kwargs['adam_eps'])
        
        self.recurrent = False
        self.rnn_keys = [ key for key,value in self.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.rnn_keys):
            self.recurrent = True 
        
        self.storage = Storage(self.kwargs['horizon'])
        if self.recurrent:
            self.storage.add_key('rnn_states')
            self.storage.add_key('next_rnn_states')
        
    def train(self):
        self.storage.placeholder()

        self.compute_advantages_and_returns()
        states, actions, log_probs_old, returns, advantages, rnn_states = self.retrieve_values_from_storage()
        
        if self.recurrent: rnn_states = self.reformat_rnn_states(rnn_states)
        
        for _ in range(self.kwargs['optimization_epochs']):
            self.optimize_model(states, actions, log_probs_old, returns, advantages, rnn_states)

        self.storage.reset()

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

    def compute_advantages_and_returns(self):
        advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32)) # TODO explain (used to be number of workers)
        returns = self.storage.V[-1].detach()
        for i in reversed(range(self.kwargs['horizon'])):
            returns = self.storage.r[i] + self.kwargs['discount'] * self.storage.non_terminal[i] * returns
            if not self.kwargs['use_gae']:
                advantages = returns - self.storage.V[i].detach()
            else:
                td_error = self.storage.r[i] + self.kwargs['discount'] * self.storage.non_terminal[i] * self.storage.V[i + 1] - self.storage.V[i]
                advantages = advantages * self.kwargs['gae_tau'] * self.kwargs['discount'] * self.storage.non_terminal[i] + td_error
            self.storage.adv[i] = advantages.detach()
            self.storage.ret[i] = returns.detach()

    def retrieve_values_from_storage(self):
        if self.recurrent:
            cat = self.storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv', 'rnn_states'])
            rnn_states = cat[-1]
            states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), cat[:-1]) 
        else:
            states, actions, log_probs_old, returns, advantages= map(lambda x: torch.cat(x, dim=0), self.storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv']) )
            rnn_states = None 

        advantages = self.standardize(advantages)
        return states, actions, log_probs_old, returns, advantages, rnn_states

    def standardize(self, x):
        return (x - x.mean()) / x.std()

    def optimize_model(self, states, actions, log_probs_old, returns, advantages, rnn_states=None):
        sampler = random_sample(np.arange(states.size(0)), self.kwargs['mini_batch_size'])
        assert( (self.recurrent and rnn_states is not None) or not(self.recurrent or rnn_states is not None) )
        if self.recurrent:
            nbr_layers_per_rnn = { k:len(rnn_states[k][0] ) for k in self.rnn_keys}
        
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_log_probs_old = log_probs_old[batch_indices].cuda() if self.kwargs['use_cuda'] else log_probs_old[batch_indices]
            sampled_returns = returns[batch_indices].cuda() if self.kwargs['use_cuda'] else returns[batch_indices]
            sampled_advantages = advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else advantages[batch_indices]

            if self.recurrent:
                sampled_rnn_states = { k: ([None]*nbr_layers_per_rnn[k] , [None]*nbr_layers_per_rnn[k]) for k in self.rnn_keys}
                for k in sampled_rnn_states:
                    for idx in range(nbr_layers_per_rnn[k]):
                        sampled_rnn_states[k][0][idx] = rnn_states[k][0][idx][batch_indices].cuda() if self.kwargs['use_cuda'] else rnn_states[k][0][idx][batch_indices]
                        sampled_rnn_states[k][1][idx] = rnn_states[k][1][idx][batch_indices].cuda() if self.kwargs['use_cuda'] else rnn_states[k][1][idx][batch_indices]
                
                prediction = self.model(sampled_states, sampled_actions, rnn_states=sampled_rnn_states)
            else:
                prediction = self.model(sampled_states, sampled_actions)
                    
            ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
            obj = ratio * sampled_advantages
            obj_clipped = ratio.clamp(1.0 - self.kwargs['ppo_ratio_clip'],
                                      1.0 + self.kwargs['ppo_ratio_clip']) * sampled_advantages
            policy_loss = -torch.min(obj, obj_clipped).mean() - self.kwargs['entropy_weight'] * prediction['entropy'].mean() # L^{clip} and L^{S} from original paper

            value_loss = 0.5 * (sampled_returns - prediction['V']).pow(2).mean()

            self.optimizer.zero_grad()
            (policy_loss + value_loss).backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
            self.optimizer.step()
