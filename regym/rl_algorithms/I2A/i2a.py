import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..networks import random_sample
from ..replay_buffers import Storage


def standardize(x):
    return (x - x.mean()) / x.std()


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
                 kwargs, 
                 latent_encoder=None):

        self.kwargs = kwargs
        self.nbr_actor = self.kwargs['nbr_actor']

        self.i2a_model = i2a_model
        self.i2a_model._reset_rnn_states()
        self.environment_model = environment_model
        self.distill_policy = distill_policy
        self.latent_encoder = latent_encoder

        self.rollout_length = kwargs['rollout_length']
        self.imagined_rollouts_per_step = kwargs['imagined_rollouts_per_step']

        self.distill_policy_update_horizon = kwargs['distill_policy_update_horizon']
        self.model_update_horizon = kwargs['model_update_horizon']
        self.environment_model_update_horizon = kwargs['environment_model_update_horizon']

        self.distill_policy_optimizer = optim.Adam(self.distill_policy.parameters(),
                                                   lr=kwargs['policies_adam_learning_rate'],
                                                   eps=kwargs['policies_adam_eps'])

        env_model_param = self.environment_model.parameters()
        '''
        # If we had the latent_encoder to the list of parameter to optimize
        # when optimizing the environment, there is a risk that they will 
        # entire in synergy that make them both collapse into something easy to predict.
        # Unless there is something to prevent the latent_encoder from collapsing, 
        # maybe a VAE-based latent_encoder could be useful.
        # Otherwise, the latent_encoder is only optimized by the RL algorithm...
        if self.kwargs['use_latent_embedding']:
            env_model_param = list(env_model_param)+list(self.latent_encoder.parameters())
        '''
        self.environment_model_optimizer = optim.Adam(env_model_param,
                                                      lr=kwargs['environment_model_learning_rate'],
                                                      eps=kwargs['environment_model_adam_eps'])

        # TODO: for the predict_network of RND to be optimized along with the model,
        # it is important to rely on the ppo algorithm instantiating the model optimizer.
        self.model_optimizer = optim.Adam(self.i2a_model.parameters(),
                                          lr=kwargs['policies_adam_learning_rate'],
                                          eps=kwargs['policies_adam_eps'])
        self.model_training_algorithm = model_training_algorithm_init_function(kwargs=kwargs,
                                                                               model=i2a_model,
                                                                               optimizer=self.model_optimizer)

        self.recurrent = self.model_training_algorithm.recurrent
        self.use_rnd = self.model_training_algorithm.use_rnd 
        self.use_cuda = kwargs['use_cuda']

        self.distill_policy_storages = None
        self.environment_model_storages = None
        self.reset_storages()
    
    def reset_storages(self):
        if self.distill_policy_storages is None and self.environment_model_storages is None:
            self.distill_policy_storages = []
            self.environment_model_storages = []
            for i in range(self.nbr_actor):
                self.distill_policy_storages.append(Storage())
                if self.i2a_model.recurrent:
                    self.distill_policy_storages[-1].add_key('rnn_states')
                    self.distill_policy_storages[-1].add_key('next_rnn_states')

                self.environment_model_storages.append(Storage())
                # Adding successive state key to compute the loss of the environment model
                self.environment_model_storages[-1].add_key('succ_s')

        for storage in self.distill_policy_storages: storage.reset()
        for storage in self.environment_model_storages: storage.reset()

        
    def retrieve_values_from_storages(self, storages, value_keys):
        full_values = [list() for _ in range(len(value_keys))]
        for storage in storages:
            # Check that there is something in the storage 
            if len(storage) == 0: continue
            cat = storage.cat(value_keys)
            for idx, (value,key) in enumerate(zip(cat,value_keys)):
                if 'rnn' in key: full_values[idx]+=value
                else: full_values[idx].append(torch.cat(value, dim=0))
        
        for idx, (values, key) in enumerate(zip(full_values,value_keys)):
            if 'rnn' in key: continue
            if 'adv' in key: values = standardize(values)
            full_values[idx] = torch.cat(values, dim=0)
            
        return (*full_values,)

    def compute_intrinsic_reward(self, state):
        return self.model_training_algorithm.compute_intrinsic_reward(state)

    def take_action(self, state):
        return self.i2a_model(state)

    def train_distill_policy(self):
        for idx, storage in enumerate(self.distill_policy_storages): 
            if len(storage) == 0: continue
            storage.placeholder()
        states, actions = self.retrieve_values_from_storages(self.distill_policy_storages, ['s', 'a'])
        rnn_states = None
        if self.i2a_model.recurrent: 
            rnn_states = self.retrieve_values_from_storages(self.distill_policy_storages, ['rnn_states'])[0]
            rnn_states = self.model_training_algorithm.reformat_rnn_states(rnn_states)

        for it in range(self.kwargs['distill_policy_optimization_epochs']):
            self.distill_policy_optimizer.zero_grad()
            distill_loss = self.compute_distill_policy_loss(states, actions, rnn_states)
            distill_loss.backward()
            nn.utils.clip_grad_norm_(self.environment_model.parameters(), self.kwargs['distill_policy_gradient_clip'])
            self.distill_policy_optimizer.step()

        for storage in self.distill_policy_storages: storage.reset()

    def train_i2a_model(self):
        self.model_training_algorithm.train()

    def train_environment_model(self):
        for idx, storage in enumerate(self.environment_model_storages): 
            if len(storage) == 0: continue
            storage.placeholder()
        states, actions, rewards, next_states = self.retrieve_values_from_storages(self.environment_model_storages, ['s', 'a', 'r', 'succ_s'])
        
        for it in range(self.kwargs['environment_model_optimization_epochs']):
            self.environment_model_optimizer.zero_grad()
            model_loss = self.compute_environment_model_loss(states, actions, rewards, next_states)
            model_loss.backward()
            nn.utils.clip_grad_norm_(self.environment_model.parameters(), self.kwargs['environment_model_gradient_clip'])
            self.environment_model_optimizer.step()

        for storage in self.environment_model_storages: storage.reset()

    def compute_environment_model_loss(self, states, actions, rewards, next_states):
        sampler = random_sample(np.arange(states.size(0)), self.kwargs['environment_model_batch_size'])
        loss = 0.0
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_rewards = rewards[batch_indices].cuda() if self.kwargs['use_cuda'] else rewards[batch_indices]
            sampled_next_sates = next_states[batch_indices].cuda() if self.kwargs['use_cuda'] else next_states[batch_indices]

            if self.kwargs['use_latent_embedding']:
                sampled_states = self.latent_encoder(sampled_states)
                sampled_next_sates = self.latent_encoder(sampled_next_sates)

            predicted_next_states, predicted_rewards = self.environment_model(sampled_states, sampled_actions)
            loss += 0.5 * (predicted_next_states - sampled_next_sates).pow(2).mean()
            loss += 0.5 * (predicted_rewards - sampled_rewards).pow(2).mean()

        return loss

    def compute_distill_policy_loss(self, states, actions, rnn_states=None):
        # Note: this formula may only work with discrete actions?
        # Formula: cross_entropy_coefficient * softmax_probabilities(actor_critic_logit) * softmax_probabilities(distil_logit)).sum(1).mean()
        nbr_layers_per_rnn = None
        if self.i2a_model.recurrent:
            nbr_layers_per_rnn = {recurrent_submodule_name: len(rnn_states[recurrent_submodule_name]['hidden'])
                                  for recurrent_submodule_name in rnn_states}

        sampler = random_sample(np.arange(states.size(0)), self.kwargs['distill_policy_batch_size'])
        loss = 0.0
        sampled_rnn_states = None
        sampled_states = None
        sampled_actions = None
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()

            if self.i2a_model.recurrent:
                sampled_rnn_states = self.model_training_algorithm.calculate_rnn_states_from_batch_indices(rnn_states, batch_indices, nbr_layers_per_rnn)

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]

            model_prediction = self.i2a_model(state=sampled_states, 
                                              action=sampled_actions,
                                              rnn_states=sampled_rnn_states)
            if self.kwargs['use_latent_embedding']:
                sampled_states = self.i2a_model.embedded_state

            distill_prediction = self.distill_policy(sampled_states,sampled_actions)
            
            loss += 0.01 * (F.softmax(model_prediction['action_logits'], dim=1).detach() * F.log_softmax(distill_prediction['action_logits'], dim=1)).sum(1).mean()

        return loss
