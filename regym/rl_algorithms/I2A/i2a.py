import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..networks import random_sample
from ..replay_buffers import Storage


class I2AAlgorithm():
    '''
    Original paper: https://arxiv.org/abs/1707.06203

    Note: The {model_training_algorithm} used to compute the loss function
          which will propagate through the :param model_free_network:,
          :param actor_critic_head: and :param rollout_encoder: needs
          to adhere to a specific function signature TODO: describe function signature
    '''

    def __init__(self, model_training_algorithm_init_function,
                 i2a_model,
                 environment_model,
                 distill_policy,
                 kwargs):

        self.i2a_model = i2a_model
        self.environment_model = environment_model
        self.distill_policy = distill_policy

        self.rollout_length = kwargs['rollout_length']
        self.imagined_rollouts_per_step = kwargs['imagined_rollouts_per_step']

        self.distill_policy_update_horizon = kwargs['distill_policy_update_horizon']
        self.model_update_horizon = kwargs['model_update_horizon']
        self.environment_model_update_horizon = kwargs['environment_model_update_horizon']

        self.distill_policy_storage = Storage(size=self.distill_policy_update_horizon)
        self.environment_model_storage = Storage(size=self.environment_model_update_horizon)

        # Adding successive state key to compute the loss of the environment model
        self.environment_model_storage.add_key('succ_s')

        self.distill_optimizer = optim.Adam(self.distill_policy.parameters(),
                                            lr=kwargs['policies_adam_learning_rate'],
                                            eps=kwargs['policies_adam_eps'])

        self.environment_model_optimizer = optim.Adam(self.environment_model.parameters(),
                                                      lr=kwargs['environment_model_learning_rate'],
                                                      eps=kwargs['environment_model_adam_eps'])

        self.model_optimizer = optim.Adam(self.i2a_model.parameters(),
                                          lr=kwargs['policies_adam_learning_rate'],
                                          eps=kwargs['policies_adam_eps'])
        self.model_training_algorithm = model_training_algorithm_init_function(kwargs=kwargs,
                                                                               model=i2a_model,
                                                                               optimizer=self.model_optimizer)

        self.use_cuda = kwargs['use_cuda']
        self.kwargs = kwargs

    def take_action(self, state):
        return self.i2a_model(state,
                              imagined_rollouts_per_step=self.imagined_rollouts_per_step,
                              rollout_length=self.rollout_length)

    def train_distill_policy(self):
        self.distill_policy_storage.placeholder()
        self.distill_policy.zero_grad()

        distill_loss = self.compute_distill_policy_loss()

        distill_loss.backward()
        nn.utils.clip_grad_norm_(self.environment_model.parameters(), self.kwargs['distill_policy_gradient_clip'])

        self.distill_policy.step()
        self.distill_policy_storage.reset()

    def train_i2a_model(self):
        self.model_training_algorithm.train()

    def train_environment_model(self):
        self.environment_model_storage.placeholder()
        self.environment_model_optimizer.zero_grad()

        model_loss = self.compute_environment_model_loss()

        model_loss.backward()
        nn.utils.clip_grad_norm_(self.environment_model.parameters(), self.kwargs['environment_model_gradient_clip'])

        self.environment_model_optimizer.step()
        self.environment_model_storage.reset()

    def compute_environment_model_loss(self): # TODO
        states, actions, rewards, next_states = map(lambda x: torch.cat(x, dim=0), self.storage.cat(['s', 'a', 'r', 'succ_s']))

        sampler = random_sample(np.arange(states.size(0)), self.kwargs['environment_model_batch_size'])
        loss = 0.0
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_next_sates = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]

            predicted_next_states, predicted_rewards = self.environment_model(sampled_states, sampled_actions)
            loss += 0.5 * (predicted_next_states - sampled_next_sates).pow(2).mean()
            loss += 0.5 * (predicted_rewards - sampled_rewards).pow(2).mean()

        return loss

    def compute_distill_policy_loss(self): # TODO
        # Note: this formula may only work with discrete actions?
        # Formula: cross_entropy_coefficient * softmax_probabilities(actor_critic_logit) * softmax_probabilities(distil_logit)).sum(1).mean()
        states, actions = map(lambda x: torch.cat(x, dim=0), self.storage.cat(['s', 'a']))

        sampler = random_sample(np.arange(states.size(0)), self.kwargs['distill_policy_batch_size'])
        loss = 0.0
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]

            distill_prediction = self.distill_policy(sampled_states,sampled_actions)
            model_prediction = self.i2a_model(sampled_states,sampled_actions)

            loss += 0.01 * (F.softmax(model_prediction['action_logits']).detach() * F.log_softmax(distill_prediction['action_logits'])).sum(1).mean()

        return loss
