from typing import Dict, List
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import regym
from regym.rl_algorithms.agents import Agent
from regym.networks import CategoricalActorCriticNet, GaussianActorCriticNet
from regym.networks import FCBody, LSTMBody, Convolutional2DBody
from regym.networks.preprocessing import turn_into_single_element_batch

from regym.rl_algorithms.PPO import PPOAlgorithm



class PPOAgent(Agent):

    def __init__(self, name, algorithm):
        self.algorithm = algorithm  # This has to go before the super initializer
        super(PPOAgent, self).__init__(name)
        self.state_preprocessing = self.algorithm.kwargs['state_preprocess']

        self.recurrent = False
        self.rnn_states = None
        self.rnn_keys = [key for key, value in self.algorithm.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.rnn_keys):
            self.recurrent = True
            self._reset_rnn_states()

    @Agent.num_actors.setter
    def num_actors(self, n):
        self._num_actors = n
        self.algorithm.storages = self.algorithm.create_storages(
            num_storages=n,
            size=(self.algorithm.horizon // self._num_actors)+1)

    def _reset_rnn_states(self):
        self.rnn_states = {k: None for k in self.rnn_keys}
        for k in self.rnn_states:
            if 'phi' in k:
                self.rnn_states[k] = self.algorithm.model.network.body.get_reset_states(cuda=self.algorithm.use_cuda)
            if 'critic' in k:
                self.rnn_states[k] = self.algorithm.model.network.critic_body.get_reset_states(cuda=self.algorithm.use_cuda)
            if 'actor' in k:
                self.rnn_states[k] = self.algorithm.model.network.actor_body.get_reset_states(cuda=self.algorithm.use_cuda)

    def _post_process(self, prediction):
        if self.recurrent:
            for k, (hs, cs) in self.rnn_states.items():
                for idx in range(len(hs)):
                    self.rnn_states[k][0][idx] = torch.Tensor(prediction['next_rnn_states'][k][0][idx].cpu())
                    self.rnn_states[k][1][idx] = torch.Tensor(prediction['next_rnn_states'][k][1][idx].cpu())

            for k, v in prediction.items():
                if isinstance(v, dict):
                    for vk, (hs, cs) in v.items():
                        for idx in range(len(hs)):
                            hs[idx] = hs[idx].detach().cpu()
                            cs[idx] = cs[idx].detach().cpu()
                        prediction[k][vk] = (hs, cs)
                else:
                    prediction[k] = v.detach().cpu()
        else:
            prediction = {k: v.detach().cpu() for k, v in prediction.items()}

        return prediction

    def _pre_process_rnn_states(self, done=False):
        if done or self.rnn_states is None: self._reset_rnn_states()
        if self.algorithm.use_cuda:
            for k, (hs, cs) in self.rnn_states.items():
                for idx in range(len(hs)):
                    self.rnn_states[k][0][idx] = self.rnn_states[k][0][idx].cuda()
                    self.rnn_states[k][1][idx] = self.rnn_states[k][1][idx].cuda()

    def handle_experience(self, s, a, r, succ_s, done, storage_idx=0):
        super(PPOAgent, self).handle_experience(s, a, r, succ_s, done)
        storage = self.algorithm.storages[storage_idx]

        non_terminal = torch.ones(1)*(1 - int(done))
        state = self.state_preprocessing(s)
        r = torch.FloatTensor([r])

        # This is not pretty, and is the remnants of
        # porting single actor PPO to multiactor
        for k in self.current_prediction.keys():
            storage.add({k: self.current_prediction[k][storage_idx]})

        storage.add({'r': r, 'non_terminal': non_terminal, 's': state})

        if (self.handled_experiences % self.algorithm.horizon) == 0:
            next_state = self.state_preprocessing(succ_s)

            if self.recurrent:
                self._pre_process_rnn_states(done=done)
                next_prediction = self.algorithm.model(next_state, rnn_states=self.rnn_states)
            else:
                next_prediction = self.algorithm.model(next_state)
            next_prediction = self._post_process(next_prediction)

            for k in next_prediction.keys():
                # Index 0 because there's a single prediction
                storage.add({k: next_prediction[k][0]})
            self.algorithm.train()
            self.handled_experiences = 0

    def handle_multiple_experiences(self, experiences: List, env_ids: List[int]):
        for (o, a, r, succ_o, done), e_i in zip(experiences, env_ids):
            self.handle_experience(o, a, r, succ_o, done, storage_idx=e_i)

    def model_free_take_action(self, state, legal_actions: List[int] = None,
                               multi_action: bool = False):
        preprocessed_state = self.state_preprocessing(state)

        self.current_prediction = self.compute_prediction(preprocessed_state,
                                                          legal_actions)
        action = self.current_prediction['a']
        if not multi_action:  # Action is a single integer
            action = np.int(action)
        if multi_action:  # Action comes from a vector env, one action per environment
            action = action.view(1, -1).squeeze(0).numpy()
        return action

    def compute_prediction(self, preprocessed_state, legal_actions) -> Dict:
        if self.recurrent:
            self._pre_process_rnn_states()
            current_prediction = self.algorithm.model(preprocessed_state,
                    rnn_states=self.rnn_states, legal_actions=legal_actions)
        else:
            current_prediction = self.algorithm.model(preprocessed_state,
                    legal_actions=legal_actions)
        return self._post_process(current_prediction)

    def clone(self, training=None):
        clone = PPOAgent(name=self.name, algorithm=copy.deepcopy(self.algorithm))
        clone.training = training

        return clone


def create_model(task: regym.environments.Task,
                 config: Dict[str, object]) -> nn.Module:
    input_dim = task.observation_dim
    if config['phi_arch'] != 'None':
        output_dim = 64  # TODO: beware of magic number
        if config['phi_arch'] == 'RNN':
            body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif config['phi_arch'] == 'MLP':
            body = FCBody(input_dim, hidden_units=(output_dim, output_dim), gate=F.leaky_relu)
        elif config['phi_arch'] == 'CNN':
            body = Convolutional2DBody(input_shape=config['preprocessed_input_dimensions'],
                                        channels=config['channels'],
                                        kernel_sizes=config['kernel_sizes'],
                                        paddings=config['paddings'],
                                        strides=config['strides'],
                                        residual_connections=config.get('residual_connections', []),
                                        use_batch_normalization=config['use_batch_normalization'])
        input_dim = output_dim
    else:
        body = None

    if config['actor_arch'] != 'None':
        output_dim = 64
        if config['actor_arch'] == 'RNN':
            actor_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif config['actor_arch'] == 'MLP':
            actor_body = FCBody(input_dim, hidden_units=(output_dim, output_dim), gate=F.leaky_relu)
    else:
        actor_body = None

    if config['critic_arch'] != 'None':
        output_dim = 64
        if config['critic_arch'] == 'RNN':
            critic_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif config['critic_arch'] == 'MLP':
            critic_body = FCBody(input_dim, hidden_units=(output_dim, output_dim), gate=F.leaky_relu)
    else:
        critic_body = None

    if task.action_type == 'Discrete' and task.observation_type == 'Discrete':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          body=body,
                                          actor_body=actor_body,
                                          critic_body=critic_body)
    if task.action_type == 'Discrete' and task.observation_type == 'Continuous':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          body=body,
                                          actor_body=actor_body,
                                          critic_body=critic_body)
    if task.action_type == 'Continuous' and task.observation_type == 'Continuous':
        model = GaussianActorCriticNet(task.observation_dim, task.action_dim,
                                       body=body,
                                       actor_body=actor_body,
                                       critic_body=critic_body)
    return model


def build_PPO_Agent(task: regym.environments.Task, config: Dict[str, object], agent_name: str) -> PPOAgent:
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for ppo agent
    :returns: PPOAgent adapted to be trained on :param: task under :param: config
    '''
    kwargs = config.copy()
    kwargs['state_preprocess'] = turn_into_single_element_batch

    model = create_model(task, config)
    model.share_memory()

    ppo_algorithm = PPOAlgorithm(kwargs, model)
    return PPOAgent(name=agent_name, algorithm=ppo_algorithm)
