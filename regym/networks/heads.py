#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from regym.networks import LeakyReLU
from regym.networks.utils import BaseNet, layer_init, tensor
from regym.networks.bodies import DummyBody


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
            q_values = self._mask_ilegal_action_logits(q_values,
                                 legal_actions, self.action_dim)
        if action is None:
            q_value, action = q_values.max(dim=1)

        probs = F.softmax(q_values, dim=-1)
        log_probs = torch.log(probs + self.EPS)
        entropy = -1. * torch.sum(probs * log_probs, dim=-1)

        return {'a': action,
                'Q': q_values,
                'entropy': entropy}


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


class CategoricalHead(nn.Module, BaseNet):
    def __init__(self, input_dim, output_dim, body: nn.Module = None):
        super().__init__()
        self.body = body
        self.fc_categorical = layer_init(nn.Linear(input_dim, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor, legal_actions: List[int] = None):
        if self.body:
            body_embedding = self.body(x)
        logits = self.fc_categorical(body_embedding)

        if legal_actions:
            logits = self._mask_ilegal_action_logits(logits, legal_actions, self.output_dim)

        probs = F.softmax(logits, dim=-1)
        log_probs = probs.log()
        entropy = -1. * torch.sum(probs * log_probs)
        action = torch.distributions.Categorical(logits=logits).sample()
        action = action.view(-1, 1)
        return {'probs': probs, 'log_probs': log_probs, 'a': action,
                'entropy': entropy}


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


class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, body, actor_body, critic_body):
        super(ActorCriticNet, self).__init__()
        if body is None: body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(body.feature_dim)
        if critic_body is None: critic_body = DummyBody(body.feature_dim)
        self.body = body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.body.parameters())


class GaussianActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 body=None,
                 actor_body=None,
                 critic_body=None):
        super(GaussianActorCriticNet, self).__init__()
        self.network = ActorCriticNet(state_dim, action_dim, body, actor_body, critic_body)
        self.std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, action=None, rnn_states=None):
        obs = tensor(obs)

        if rnn_states is not None and 'phi_arch' in rnn_states:
            phi, rnn_states['phi_arch'] = self.network.body( (obs, rnn_states['phi_arch']) )
        else:
            phi = self.network.body(obs)

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
                    'V': v,
                    'rnn_states': rnn_states}
        else:
            return {'a': action,
                    'log_pi_a': log_prob,
                    'ent': entropy,
                    'V': v}


class CategoricalActorCriticNet(nn.Module, BaseNet):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 critic_gate_fn: Optional[str] = None,
                 body: nn.Module = None,
                 actor_body: nn.Module = None,
                 critic_body: nn.Module = None):
        BaseNet.__init__(self)
        super(CategoricalActorCriticNet, self).__init__()
        self.action_dim = action_dim

        if critic_gate_fn:
            gating_fns = {'tanh': nn.functional.tanh}
            self.critic_gate_fn = gating_fns[critic_gate_fn]
        else: self.critic_gate_fn = None

        self.network = ActorCriticNet(state_dim, action_dim, body, actor_body, critic_body)

    # TODO: type hint rnn_states
    def forward(self, obs: torch.Tensor, action: int = None, rnn_states=None,
                legal_actions: List[int] = None):
        if rnn_states is not None:
            next_rnn_states = {k: None for k in rnn_states}

        if rnn_states is not None and 'phi_arch' in rnn_states:
            phi, next_rnn_states['phi_arch'] = self.network.body( (obs, rnn_states['phi_arch']) )
        else:
            phi = self.network.body(obs)

        if rnn_states is not None and 'actor_arch' in rnn_states:
            phi_a, next_rnn_states['actor_arch'] = self.network.actor_body( (phi, rnn_states['actor_arch']) )
        else:
            phi_a = self.network.actor_body(phi)

        if rnn_states is not None and 'critic_arch' in rnn_states:
            phi_v, next_rnn_states['critic_arch'] = self.network.critic_body( (phi, rnn_states['critic_arch']) )
        else:
            phi_v = self.network.critic_body(phi)

        logits = self.network.fc_action(phi_a)
        if legal_actions:
            logits = self._mask_ilegal_action_logits(logits, legal_actions, self.action_dim)
        # Size: batch x action_dim
        v = self.network.fc_critic(phi_v)
        if self.critic_gate_fn: v = self.critic_gate_fn(v)
        # Size: batch x 1

        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            # Size: batch x 1
            action = dist.sample()
            action = action.view(-1, 1)
        log_prob = dist.log_prob(action).unsqueeze(-1)
        # estimates the log likelihood of each action against each batched distributions... : batch x batch x 1
        log_prob = torch.cat([log_prob[idx][idx].view((1,-1)) for idx in range(log_prob.size(0))], dim=0)
        # retrieve the log likelihood of each batched action against its relevant batched distribution: batch x 1
        entropy = dist.entropy().unsqueeze(-1)
        # retrieve the the entropy of each batched distribution: batch x 1

        prediction = {'a': action,
                      'log_pi_a': log_prob,
                      'entropy': entropy,
                      'V': v,
                      'probs': dist.probs}
        if rnn_states is not None: prediction.update({'rnn_states': rnn_states, 'next_rnn_states': next_rnn_states})
        return prediction


class PolicyInferenceActorCriticNet(nn.Module, BaseNet):
    '''
    Inspired from A3C-AMF architecture:
        'Agent Modeling as Auxiliary Task for Deep Reinforcement Learning'
        Pablo Hernandez-Leal et al. (arXiv:1907.09597)
    Originally suggested as DPIQN architecture in:
        'A Deep Policy Inference Q-Network for Multi-Agent Systems'
        Zhang-Wei Hong et al (ariv:1712.07893)
    '''
    def __init__(self,
                 num_policies: int,
                 num_actions: int,
                 feature_extractor: BaseNet,
                 policy_inference_body: BaseNet,
                 actor_critic_body: BaseNet):
        '''
        TODO
        '''
        BaseNet.__init__(self)
        super().__init__()

        self.num_policies = num_policies
        # NOTE: all three
        # (policy_inference_body feature_dim, actor_critic_body feature_dim, actor_critic_head input_dim)
        # should be the same!

        self.feature_extractor = feature_extractor

        self.policy_inference_body = policy_inference_body
        self.actor_critic_body = actor_critic_body

        self.policy_inference_heads = nn.ModuleList([
                CategoricalHead(
                    policy_inference_body.feature_dim, num_actions)
                for _ in range(num_policies)])

        self.actor_critic_head = CategoricalActorCriticNet(
                state_dim=self.actor_critic_body.feature_dim,
                action_dim=num_actions)

    def forward(self, x: torch.Tensor, legal_actions: List[int] = None):
        feature_embeddings = self.feature_extractor(x)

        # Opponent modelling branch
        policy_type_embedding = self.policy_inference_body(feature_embeddings)
        # How about adding a non-linearity here?
        policy_inference_predictions = map(lambda head: head(policy_type_embedding),
                                           self.policy_inference_heads)

        # Actor critic branch
        actor_critic_body_embedding = self.actor_critic_body(feature_embeddings)
        # element wise vector multiplication
        actor_critic_head_input = policy_type_embedding * actor_critic_body_embedding

        actor_critic_prediction = self.actor_critic_head(
                actor_critic_head_input, legal_actions=legal_actions)

        policy_inference_predictions_dict = {
                f'policy_{i}': prediction
                for i, prediction
                in zip(range(self.num_policies), policy_inference_predictions)}

        # Merging 2 dicts together
        aggregated_predictions = {
                **policy_inference_predictions_dict, **actor_critic_prediction}

        # TODO: add actor_critic head
        return aggregated_predictions
