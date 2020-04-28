#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from regym.rl_algorithms.networks import LeakyReLU
from regym.rl_algorithms.networks.utils import BaseNet, layer_init, tensor
from regym.rl_algorithms.networks.bodies import DummyBody


class CategoricalDQNet(nn.Module, BaseNet):

    def __init__(self,
                 body: nn.Module,
                 action_dim: int,
                 actfn=LeakyReLU,
                 use_cuda=False):
        BaseNet.__init__(self)
        super(CategoricalDQNet, self).__init__()

        self.body = body
        self.action_dim = action_dim
        self.use_cuda = use_cuda

        self.actfn = actfn

        body_output_features = self.body.feature_dim
        self.qsa = nn.Linear(body_output_features, self.action_dim)

        if self.use_cuda:
            self = self.cuda()

    def forward(self, x: torch.Tensor, action: torch.Tensor = None,
                legal_actions: List[int] = None):
        # Forward pass till last layer
        x = self.body(x)
        # Q values for all actions
        q_values = self.qsa(x)

        # TODO: check if this works
        if legal_actions is not None:
            q_values = self._mask_ilegal_action_logits(q_values, legal_actions)
        if action is None:
            q_value, action = q_values.max(dim=1)

        probs = F.softmax(q_values, dim=-1)
        log_probs = torch.log(probs + self.EPS)
        entropy = -1. * torch.sum(probs * log_probs, dim=-1)

        return {'a': action,
                'Q': q_values,
                'entropy': entropy}

    def _mask_ilegal_action_logits(self, logits: torch.Tensor, legal_actions: List[int]):
        '''
        TODO: document
        '''
        illegal_action_mask = torch.tensor([float(i not in legal_actions)
                                            for i in range(self.action_dim)])
        illegal_logit_penalties = illegal_action_mask * self.ILLEGAL_ACTIONS_LOGIT_PENALTY
        masked_logits = logits + illegal_logit_penalties
        return masked_logits


class CategoricalDuelingDQNet(nn.Module, BaseNet):

    def __init__(self,
                 body: nn.Module,
                 action_dim: int,
                 actfn=LeakyReLU):
        BaseNet.__init__(self)
        super(CategoricalDuelingDQNet, self).__init__()
        self.action_dim = action_dim

        self.body = body

        self.value = layer_init(nn.Linear(body.feature_dim, 1))
        self.advantage = layer_init(nn.Linear(body.feature_dim, action_dim))

    def forward(self, x, action: torch.Tensor = None,
                legal_actions: List[int] = None):
        x = self.body(tensor(x))
        V = self.value(x)
        A = self.advantage(x)

        Q = V.expand_as(A) + (A - A.mean(1, keepdim=True))

        probs = F.softmax(Q, dim=-1)
        log_probs = torch.log(probs + self.EPS)
        entropy = -1. * torch.sum(probs * log_probs, dim=-1)

        return {'V': V, 'A': A, 'Q': Q, 'a': Q.max(dim=1)[1],
                'entropy': entropy}


class CategoricalNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_atoms, body):
        super(CategoricalNet, self).__init__()
        self.fc_categorical = layer_init(nn.Linear(body.feature_dim, action_dim * num_atoms))
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.body = body

    def forward(self, x):
        phi = self.body(tensor(x))
        pre_prob = self.fc_categorical(phi).view((-1, self.action_dim, self.num_atoms))
        prob = F.softmax(pre_prob, dim=-1)
        log_prob = F.log_softmax(pre_prob, dim=-1)
        return prob, log_prob


class QuantileNet(nn.Module, BaseNet):
    def __init__(self, action_dim, num_quantiles, body):
        super(QuantileNet, self).__init__()
        self.fc_quantiles = layer_init(nn.Linear(body.feature_dim, action_dim * num_quantiles))
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        quantiles = self.fc_quantiles(phi)
        quantiles = quantiles.view((-1, self.action_dim, self.num_quantiles))
        return quantiles


class OptionCriticNet(nn.Module, BaseNet):
    def __init__(self, body, action_dim, num_options):
        super(OptionCriticNet, self).__init__()
        self.fc_q = layer_init(nn.Linear(body.feature_dim, num_options))
        self.fc_pi = layer_init(nn.Linear(body.feature_dim, num_options * action_dim))
        self.fc_beta = layer_init(nn.Linear(body.feature_dim, num_options))
        self.num_options = num_options
        self.action_dim = action_dim
        self.body = body
        self.to(Config.DEVICE)

    def forward(self, x):
        phi = self.body(tensor(x))
        q = self.fc_q(phi)
        beta = F.sigmoid(self.fc_beta(phi))
        pi = self.fc_pi(phi)
        pi = pi.view(-1, self.num_options, self.action_dim)
        log_pi = F.log_softmax(pi, dim=-1)
        return q, beta, log_pi


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, phi_body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())


class DeterministicActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.actor_opt = actor_opt_fn(self.network.actor_params + self.network.phi_params)
        self.critic_opt = critic_opt_fn(self.network.critic_params + self.network.phi_params)
        self.to(Config.DEVICE)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.network.phi_body(obs)

    def actor(self, phi):
        return F.tanh(self.network.fc_action(self.network.actor_body(phi)))

    def critic(self, phi, a):
        return self.network.fc_critic(self.network.critic_body(phi, a))


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)
        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, action=None, rnn_states=None):
        obs = tensor(obs)

        if rnn_states is not None and 'phi_arch' in rnn_states:
            phi, rnn_states['phi_arch'] = self.network.phi_body( (obs, rnn_states['phi_arch']) )
        else:
            phi = self.network.phi_body(obs)

        if rnn_states is not None and 'actor_arch' in rnn_states:
            phi_a, rnn_states['actor_arch'] = self.network.actor_body( (phi, rnn_states['actor_arch']) )
        else:
            phi_a = self.network.actor_body(phi)

        if rnn_states is not None and 'critic_arch' in rnn_states:
            phi_v, rnn_states['critic_arch'] = self.network.critic_body( (phi, rnn_states['critic_arch']) )
        else:
            phi_v = self.network.critic_body(phi)

        mean = F.tanh(self.network.fc_action(phi_a))
        v = self.network.fc_critic(phi_v)
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)

        if rnn_states is not None:
            return {'a': action,
                    'log_pi_a': log_prob,
                    'ent': entropy,
                    'v': v,
                    'rnn_states': rnn_states}
        else:
            return {'a': action,
                    'log_pi_a': log_prob,
                    'ent': entropy,
                    'v': v}


class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 phi_body: nn.Module = None,
                 actor_body: nn.Module = None,
                 critic_body: nn.Module = None):
        BaseNet.__init__(self)
        super(CategoricalActorCriticNet, self).__init__()
        self.action_dim = action_dim
        self.network = ActorCriticNet(state_dim, action_dim, phi_body, actor_body, critic_body)

    # TODO: type hint rnn_states
    def forward(self, obs: torch.Tensor, action: int = None, rnn_states=None,
                legal_actions: List[int] = None):
        if rnn_states is not None:
            next_rnn_states = {k: None for k in rnn_states}

        if rnn_states is not None and 'phi_arch' in rnn_states:
            phi, next_rnn_states['phi_arch'] = self.network.phi_body( (obs, rnn_states['phi_arch']) )
        else:
            phi = self.network.phi_body(obs)

        if rnn_states is not None and 'actor_arch' in rnn_states:
            phi_a, next_rnn_states['actor_arch'] = self.network.actor_body( (phi, rnn_states['actor_arch']) )
        else:
            phi_a = self.network.actor_body(phi)

        if rnn_states is not None and 'critic_arch' in rnn_states:
            phi_v, next_rnn_states['critic_arch'] = self.network.critic_body( (phi, rnn_states['critic_arch']) )
        else:
            phi_v = self.network.critic_body(phi)

        logits = self.network.fc_action(phi_a)
        if legal_actions is not None:
            logits = self._mask_ilegal_action_logits(logits, legal_actions)
        # batch x action_dim
        v = self.network.fc_critic(phi_v)
        # batch x 1

        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample(sample_shape=(logits.size(0),))
            # batch x 1
        log_prob = dist.log_prob(action).unsqueeze(-1)
        # estimates the log likelihood of each action against each batched distributions... : batch x batch x 1
        log_prob = torch.cat([log_prob[idx][idx].view((1,-1)) for idx in range(log_prob.size(0))], dim=0)
        # retrieve the log likelihood of each batched action against its relevant batched distribution: batch x 1
        entropy = dist.entropy().unsqueeze(-1)
        # retrieve the the entropy of each batched distribution: batch x 1

        prediction = {'a': action,
                      'log_pi_a': log_prob,
                      'entropy': entropy,
                      'v': v,
                      'probs': dist.probs}
        if rnn_states is not None: prediction.update({'rnn_states': rnn_states, 'next_rnn_states': next_rnn_states})
        return prediction

    def _mask_ilegal_action_logits(self, logits: torch.Tensor, legal_actions: List[int]):
        '''
        TODO: document 
        '''
        illegal_action_mask = torch.tensor([float(i not in legal_actions)
                                            for i in range(self.action_dim)])
        illegal_logit_penalties = illegal_action_mask * self.ILLEGAL_ACTIONS_LOGIT_PENALTY
        masked_logits = logits + illegal_logit_penalties
        return masked_logits
