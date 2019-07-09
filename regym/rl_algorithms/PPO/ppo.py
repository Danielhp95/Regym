from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..networks import random_sample
from ..replay_buffers import Storage
from . import ppo_loss, rnd_loss


class PPOAlgorithm():

    def __init__(self, kwargs, model, optimizer=None, target_intr_model=None, predict_intr_model=None):
        '''
        TODO specify which values live inside of kwargs
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
        self.nbr_actor = self.kwargs['nbr_actor']
        self.use_rnd = False
        if target_intr_model is not None and predict_intr_model is not None:
            self.use_rnd = True
            self.target_intr_model = target_intr_model
            self.predict_intr_model = predict_intr_model
            self.states_mean = 0.0
            self.states_std = 1.0
            self.int_reward_mean = 0.0
            self.int_reward_std = 1.0
            self.running_counter_intrinsic_reward = 0
            self.update_period_intrinsic_reward = self.kwargs['rnd_update_period_running_meanstd_int_reward']

        self.model = model
        if self.kwargs['use_cuda']:
            self.model = self.model.cuda()
            if self.use_rnd:
                self.target_intr_model = self.target_intr_model.cuda()
                self.predict_intr_model = self.predict_intr_model.cuda()
        

        if optimizer is None:
            parameters = self.model.parameters()
            if self.use_rnd: parameters = list(parameters)+list(self.predict_intr_model.parameters())
            self.optimizer = optim.Adam(parameters, lr=kwargs['learning_rate'], eps=kwargs['adam_eps'])
        else: self.optimizer = optimizer

        self.recurrent = False
        # TECHNICAL DEBT: check for recurrent property by looking at the modules in the model rather than relying on the kwargs that may contain
        # elements that do not concern the model trained by this algorithm, given that it is now use-able inside I2A...
        self.recurrent_nn_submodule_names = [hyperparameter for hyperparameter, value in self.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.recurrent_nn_submodule_names): self.recurrent = True

        self.storages = None
        self.reset_storages()

    def reset_storages(self):
        if self.storages is not None:
            for storage in self.storages: storage.reset()

        self.storages = []
        for i in range(self.nbr_actor):
            self.storages.append(Storage())
            if self.recurrent:
                self.storages[-1].add_key('rnn_states')
                self.storages[-1].add_key('next_rnn_states')
            if self.use_rnd:
                self.storages[-1].add_key('int_r')
                self.storages[-1].add_key('int_v')
                self.storages[-1].add_key('int_ret')
                self.storages[-1].add_key('int_adv')
                self.storages[-1].add_key('target_int_f')

    def train(self):
        for idx, storage in enumerate(self.storages): 
            # Check that there is something in the storage 
            if len(storage) == 0: continue
            storage.placeholder()
            self.compute_advantages_and_returns(storage_idx=idx)
            if self.use_rnd: self.compute_int_advantages_and_int_returns(storage_idx=idx)

        states, actions, log_probs_old, returns, advantages, ext_v_old, int_v_old, int_returns, int_advantages, target_random_features, rnn_states = self.retrieve_values_from_storages()

        if self.recurrent: rnn_states = self.reformat_rnn_states(rnn_states)

        for _ in range(self.kwargs['optimization_epochs']):
            self.optimize_model(states, actions, log_probs_old, returns, advantages, ext_v_old, int_v_old, int_returns, int_advantages, target_random_features, rnn_states)

        if self.use_rnd: 
            # Compute states_mean and states_std to be reused on the next optimization call:
            self.compute_states_mean_std(states)

        self.reset_storages()

    def reformat_rnn_states(self, rnn_states):
        '''
        This function reformats the :param rnn_states: into 
        a dict of dict of list of batched rnn_states.
        :param rnn_states: list of dict of dict of list: each element is an rnn_state where:
            - the first dictionnary has the name of the recurrent module in the architecture
              as keys.
            - the second dictionnary has the keys 'hidden', 'cell'.
            - the items of this second dictionnary are lists of actual hidden/cell states for the LSTMBody.
        '''
        reformated_rnn_states = {k: {'hidden': [list()], 'cell': [list()]} for k in rnn_states[0]}
        for rnn_state in rnn_states:
            for k in rnn_state:
                hstates, cstates = rnn_state[k]['hidden'], rnn_state[k]['cell']
                for idx_layer, (h, c) in enumerate(zip(hstates, cstates)):
                    reformated_rnn_states[k]['hidden'][0].append(h)
                    reformated_rnn_states[k]['cell'][0].append(c)
        for k in reformated_rnn_states:
            hstates, cstates = reformated_rnn_states[k]['hidden'], reformated_rnn_states[k]['cell']
            hstates = torch.cat(hstates[0], dim=0)
            cstates = torch.cat(cstates[0], dim=0)
            reformated_rnn_states[k] = {'hidden': [hstates], 'cell': [cstates]}
        return reformated_rnn_states

    def compute_advantages_and_returns(self, storage_idx):
        advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32)) # TODO explain (used to be number of workers)
        returns = self.storages[storage_idx].v[-1].detach()
        for i in reversed(range(len(self.storages[storage_idx])-1)):
            returns = self.storages[storage_idx].r[i] + self.kwargs['discount'] * self.storages[storage_idx].non_terminal[i] * returns
            if not self.kwargs['use_gae']:
                advantages = returns - self.storages[storage_idx].v[i].detach()
            else:
                td_error = self.storages[storage_idx].r[i] 
                td_error = td_error + self.kwargs['discount'] * self.storages[storage_idx].non_terminal[i] * self.storages[storage_idx].v[i + 1].detach() 
                td_error = td_error - self.storages[storage_idx].v[i].detach()
                advantages = advantages * self.kwargs['gae_tau'] * self.kwargs['discount'] * self.storages[storage_idx].non_terminal[i] + td_error
            self.storages[storage_idx].adv[i] = advantages.detach()
            self.storages[storage_idx].ret[i] = returns.detach()

    def compute_int_advantages_and_int_returns(self, storage_idx):
        '''
        Compute intrinsic returns and advantages from normalized intrinsic rewards.
        Indeed, int_r values in storages have been normalized upon computation.
        At computation-time, updates of the running mean and std are performed too.
        '''
        norm_int_r = self.normalize_int_rewards(storage_idx)
        int_advantages = torch.from_numpy(np.zeros((1, 1), dtype=np.float32))
        int_returns = self.storages[storage_idx].int_v[-1].detach()
        for i in reversed(range(len(self.storages[storage_idx])-1)):
            #int_returns = self.storages[storage_idx].int_r[i] + self.kwargs['intrinsic_discount'] * self.storages[storage_idx].non_terminal[i] * int_returns
            int_returns = norm_int_r[i] + self.kwargs['intrinsic_discount'] * self.storages[storage_idx].non_terminal[i] * int_returns
            if not self.kwargs['use_gae']:
                int_advantages = int_returns - self.storages[storage_idx].int_v[i].detach()
            else:
                #td_error = self.storages[storage_idx].int_r[i] 
                td_error = norm_int_r[i] 
                td_error = td_error + self.kwargs['intrinsic_discount'] * self.storages[storage_idx].non_terminal[i] * self.storages[storage_idx].int_v[i + 1].detach() 
                td_error = td_error - self.storages[storage_idx].int_v[i].detach()
                int_advantages = int_advantages * self.kwargs['gae_tau'] * self.kwargs['intrinsic_discount'] * self.storages[storage_idx].non_terminal[i] + td_error
            self.storages[storage_idx].int_adv[i] = int_advantages.detach()
            self.storages[storage_idx].int_ret[i] = int_returns.detach()

    def normalize_int_rewards(self, storage_idx):
        normalized_int_rewards = []
        for i in range(len(self.storages[storage_idx])):
            normalized_int_rewards.append(self.storages[storage_idx].int_r[i] / (self.int_reward_std+1e-8))
        return normalized_int_rewards

    def retrieve_values_from_storages(self):
        full_states = []
        full_actions = []
        full_log_probs_old = []
        full_returns = []
        full_advantages = []
        full_rnn_states = None
        full_ext_v_old = None
        full_int_v_old = None
        full_int_returns = None
        full_int_advantages = None
        full_target_random_features = None
        if self.use_rnd:
            full_int_returns = []
            full_int_advantages = []
            full_ext_v_old = []
            full_int_v_old = []
            full_target_random_features = []
        if self.recurrent:
            full_rnn_states = []
            
        for storage in self.storages:
            # Check that there is something in the storage 
            if len(storage) == 0: continue
            cat = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv'])
            states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), cat)
            full_states.append(states)
            full_actions.append(actions)
            full_log_probs_old.append(log_probs_old)
            full_returns.append(returns)
            full_advantages.append(advantages)
            if self.use_rnd:
                cat = storage.cat(['v', 'int_v', 'int_ret', 'int_adv', 'target_int_f'])
                ext_v_old, int_v_old, int_returns, int_advantages, target_random_features = map(lambda x: torch.cat(x, dim=0), cat)
                full_int_returns.append(int_returns)
                full_int_advantages.append(int_advantages)
                full_ext_v_old.append(ext_v_old)
                full_int_v_old.append(int_v_old)
                full_target_random_features.append(target_random_features)
            if self.recurrent:
                rnn_states = storage.cat(['rnn_states'])[0]
                full_rnn_states += rnn_states
            
        full_states = torch.cat(full_states, dim=0)
        full_actions = torch.cat(full_actions, dim=0)
        full_log_probs_old = torch.cat(full_log_probs_old, dim=0)
        full_returns = torch.cat(full_returns, dim=0)
        full_advantages = torch.cat(full_advantages, dim=0)
        full_advantages = self.standardize(full_advantages)
        if self.use_rnd:
            full_ext_v_old = torch.cat(full_ext_v_old, dim=0)
            full_int_v_old = torch.cat(full_int_v_old, dim=0)
            full_int_returns = torch.cat(full_int_returns, dim=0)
            full_int_advantages = torch.cat(full_int_advantages, dim=0)
            full_target_random_features = torch.cat(full_target_random_features, dim=0)
            full_int_advantages = self.standardize(full_int_advantages)
            
        return full_states, full_actions, full_log_probs_old, full_returns, full_advantages, full_ext_v_old, full_int_v_old, full_int_returns, full_int_advantages, full_target_random_features, full_rnn_states

    def standardize(self, x):
        return (x - x.mean()) / x.std()

    def compute_intrinsic_reward(self, states):
        normalized_states = (states-self.states_mean) / self.states_std 
        if self.kwargs['use_cuda']: normalized_states = normalized_states.cuda()
        pred_features = self.predict_intr_model(normalized_states)
        target_features = self.target_intr_model(normalized_states)
        int_reward = torch.nn.functional.mse_loss(target_features,pred_features)
        #int_reward = torch.clamp(int_reward, -1, 1)
        int_reward = int_reward.detach().cpu()
        self.update_int_reward_mean_std(int_reward)
        
        #int_r = int_reward / (self.int_reward_std+1e-8)
        #return int_r, target_features.detach().cpu()

        return int_reward, target_features.detach().cpu()

    def compute_states_mean_std(self,states):
        self.states_mean = states.mean()
        self.states_std = states.std() + 1e-5

    def update_int_reward_mean_std(self, unnormalized_ir):
        rmean = self.int_reward_mean
        rstd = self.int_reward_std
        rc = self.running_counter_intrinsic_reward

        self.running_counter_intrinsic_reward += 1
        
        self.int_reward_mean = (self.int_reward_mean*rc+unnormalized_ir)/self.running_counter_intrinsic_reward
        self.int_reward_std = np.sqrt( ( np.power(self.int_reward_std,2)*rc+np.power(unnormalized_ir-rmean, 2) ) / self.running_counter_intrinsic_reward )
        
        if self.running_counter_intrinsic_reward >= self.update_period_intrinsic_reward:
          self.running_counter_intrinsic_reward = 0

    def optimize_model(self, states, actions, log_probs_old, returns, advantages, ext_v_old, int_v_old, int_returns, int_advantages, target_random_features, rnn_states=None):
        # What is this: create dictionary to store length of each part of the recurrent submodules of the current model
        nbr_layers_per_rnn = None
        if self.recurrent:
            nbr_layers_per_rnn = {recurrent_submodule_name: len(rnn_states[recurrent_submodule_name]['hidden'])
                                  for recurrent_submodule_name in rnn_states}

        sampler = random_sample(np.arange(advantages.size(0)), self.kwargs['mini_batch_size'])
        for batch_indices in sampler:
            batch_indices = torch.from_numpy(batch_indices).long()
            sampled_rnn_states = None
            if self.recurrent:
                sampled_rnn_states = self.calculate_rnn_states_from_batch_indices(rnn_states, batch_indices, nbr_layers_per_rnn)

            sampled_states = states[batch_indices].cuda() if self.kwargs['use_cuda'] else states[batch_indices]
            sampled_actions = actions[batch_indices].cuda() if self.kwargs['use_cuda'] else actions[batch_indices]
            sampled_log_probs_old = log_probs_old[batch_indices].cuda() if self.kwargs['use_cuda'] else log_probs_old[batch_indices]
            sampled_returns = returns[batch_indices].cuda() if self.kwargs['use_cuda'] else returns[batch_indices]
            sampled_advantages = advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else advantages[batch_indices]
                
            sampled_states = sampled_states.detach()
            sampled_actions = sampled_actions.detach()
            sampled_log_probs_old = sampled_log_probs_old.detach()
            sampled_returns = sampled_returns.detach()
            sampled_advantages = sampled_advantages.detach()

            if self.use_rnd:
                sampled_ext_v_old = ext_v_old[batch_indices].cuda() if self.kwargs['use_cuda'] else ext_v_old[batch_indices]
                sampled_int_v_old = int_v_old[batch_indices].cuda() if self.kwargs['use_cuda'] else int_v_old[batch_indices]
                sampled_int_returns = int_returns[batch_indices].cuda() if self.kwargs['use_cuda'] else int_returns[batch_indices]
                sampled_int_advantages = int_advantages[batch_indices].cuda() if self.kwargs['use_cuda'] else int_advantages[batch_indices]
                sampled_target_random_features = target_random_features[batch_indices].cuda() if self.kwargs['use_cuda'] else target_random_features[batch_indices]
                sampled_ext_v_old = sampled_ext_v_old.detach()
                sampled_int_v_old = sampled_int_v_old.detach()
                sampled_int_returns = sampled_int_returns.detach()
                sampled_int_advantages = sampled_int_advantages.detach()
                sampled_target_random_features = sampled_target_random_features.detach()

            self.optimizer.zero_grad()
            if self.use_rnd:
                loss = rnd_loss.compute_loss(sampled_states, 
                                             sampled_actions, 
                                             sampled_log_probs_old,
                                             ext_v_old=sampled_ext_v_old,
                                             int_v_old=sampled_int_v_old,
                                             ext_returns=sampled_returns, 
                                             ext_advantages=sampled_advantages,
                                             int_returns=sampled_int_returns, 
                                             int_advantages=sampled_int_advantages, 
                                             target_random_features=sampled_target_random_features,
                                             states_mean=self.states_mean, 
                                             states_std=self.states_std,
                                             rnn_states=sampled_rnn_states,
                                             ratio_clip=self.kwargs['ppo_ratio_clip'], 
                                             entropy_weight=self.kwargs['entropy_weight'],
                                             model=self.model,
                                             pred_intr_model=self.predict_intr_model,
                                             intrinsic_reward_ratio=self.kwargs['rnd_loss_int_ratio'] )
            else:
                loss = ppo_loss.compute_loss(sampled_states, sampled_actions, sampled_log_probs_old,
                                             sampled_returns, sampled_advantages, 
                                             rnn_states=sampled_rnn_states,
                                             ratio_clip=self.kwargs['ppo_ratio_clip'], entropy_weight=self.kwargs['entropy_weight'],
                                             model=self.model)

            loss.backward(retain_graph=False)
            #loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['gradient_clip'])
            self.optimizer.step()

    def calculate_rnn_states_from_batch_indices(self, rnn_states, batch_indices, nbr_layers_per_rnn):
        sampled_rnn_states = {k: {'hidden': [None]*nbr_layers_per_rnn[k], 'cell': [None]*nbr_layers_per_rnn[k]} for k in rnn_states}
        for recurrent_submodule_name in sampled_rnn_states:
            for idx in range(nbr_layers_per_rnn[recurrent_submodule_name]):
                sampled_rnn_states[recurrent_submodule_name]['hidden'][idx] = rnn_states[recurrent_submodule_name]['hidden'][idx][batch_indices].cuda() if self.kwargs['use_cuda'] else rnn_states[recurrent_submodule_name]['hidden'][idx][batch_indices]
                sampled_rnn_states[recurrent_submodule_name]['cell'][idx]   = rnn_states[recurrent_submodule_name]['cell'][idx][batch_indices].cuda() if self.kwargs['use_cuda'] else rnn_states[recurrent_submodule_name]['cell'][idx][batch_indices]
        return sampled_rnn_states

    @staticmethod
    def check_mandatory_kwarg_arguments(kwargs: dict):
        '''
        Checks that all mandatory hyperparameters are present
        inside of dictionary :param kwargs:

        :param kwargs: Dictionary of hyperparameters
        '''
        # Future improvement: add a condition to check_kwarg (discount should be between (0:1])
        keywords = ['horizon', 'discount', 'use_gae', 'gae_tau', 'use_cuda',
                    'entropy_weight', 'gradient_clip', 'optimization_epochs',
                    'mini_batch_size', 'ppo_ratio_clip', 'learning_rate', 'adam_eps']

        def check_kwarg_and_condition(keyword, kwargs):
            if keyword not in kwargs:
                raise ValueError(f"Keyword: '{keyword}' not found in kwargs")
        for keyword in keywords: check_kwarg_and_condition(keyword, kwargs)
