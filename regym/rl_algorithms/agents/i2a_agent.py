from typing import Dict, List
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from regym.rl_algorithms.networks import ResizeCNNPreprocessFunction
from regym.rl_algorithms.I2A import I2AAlgorithm, ImaginationCore, EnvironmentModel, AutoEncoderEnvironmentModel, RolloutEncoder, I2AModel
from regym.rl_algorithms.networks import CategoricalActorCriticNet, FCBody, LSTMBody, ConvolutionalBody, choose_architecture


class I2AAgent():
    '''
    Agent which acts on an environment to collect environment signals
    to improve an underlying policy model.
    '''
    def __init__(self, name: str, algorithm: I2AAlgorithm, preprocess_function,
                 rnn_keys: List[str], use_cuda: bool):
        '''
        :param name: String identifier for the agent
        :param algorithm: Reinforcement Learning algorithm used to update the agent's policy.
                          Contains the agent's policy, represented as a neural network.
        :param preprocess_function: Function which preprocesses the states before
                                    being handed to the algorithm
        :param rnn_keys:
        :param use_cuda:
        '''
        self.name = name
        self.algorithm = algorithm
        self.use_rnd = self.algorithm.use_rnd

        self.training = True
        self.use_cuda = use_cuda
        self.preprocess_function = preprocess_function

        self.handled_experiences = 0
        self.save_path = None 
        self.episode_count = 0

        self.nbr_actor = self.algorithm.kwargs['nbr_actor']
        self.previously_done_actors = [False]*self.nbr_actor

        # Current_prediction stores information
        # from the last action that was taken
        self.current_prediction: Dict[str, object]

        self.recurrent = False
        self.rnn_keys = rnn_keys
        if len(self.rnn_keys):
            self.recurrent = True
    
    def get_intrinsic_reward(self, actor_idx):
        if len(self.algorithm.model_training_algorithm.storages[actor_idx].int_r):
            return self.algorithm.model_training_algorithm.storages[actor_idx].int_r[-1] / (self.algorithm.model_training_algorithm.int_reward_std+1e-8)
        else:
            return 0.0

    @property
    def rnn_states(self):
        return self.algorithm.i2a_model.rnn_states

    def set_nbr_actor(self, nbr_actor):
        if nbr_actor != self.nbr_actor:
            self.nbr_actor = nbr_actor
            self.algorithm.kwargs['nbr_actor'] = self.nbr_actor
            self.algorithm.reset_storages()

    def reset_actors(self):
        '''
        In case of a multi-actor process, this function is called to reset
        the actors' internal values.
        '''
        self.previously_done_actors = [False]*self.nbr_actor
        if self.recurrent:
            self._reset_rnn_states()

    def _reset_rnn_states(self):
        self.algorithm.i2a_model._reset_rnn_states()

    def update_actors(self, batch_idx):
        '''
        In case of a multi-actor process, this function is called to handle
        the (dynamic) number of environment that are being used.
        More specifically, it regularizes the rnn_states when
        an actor's episode ends.
        It is assumed that update can only go by removing stuffs...
        Indeed, actors all start at the same time, and thus self.reset_actors()
        ought to be called at that time.
        Note: since the number of environment that are running changes, 
        the size of the rnn_states on the batch dimension will change too.
        Therefore, we cannot identify an rnn_state by the actor/environment index.
        Thus, it is a batch index that is requested, that would truly be in touch
        with the current batch dimension.
        :param batch_idx: index of the actor whose episode just finished.
        '''
        if self.recurrent:
            self.remove_from_rnn_states(batch_idx=batch_idx)

    def remove_from_rnn_states(self, batch_idx):
        '''
        Remove a row(=batch) of data from the rnn_states.
        :param batch_idx: index on the batch dimension that specifies which row to remove.
        '''
        for recurrent_submodule_name in self.rnn_states:
            if self.rnn_states[recurrent_submodule_name] is None: continue
            for idx in range(len(self.rnn_states[recurrent_submodule_name]['hidden'])):
                self.rnn_states[recurrent_submodule_name]['hidden'][idx] = torch.cat(
                    [self.rnn_states[recurrent_submodule_name]['hidden'][idx][:batch_idx,...], 
                     self.rnn_states[recurrent_submodule_name]['hidden'][idx][batch_idx+1:,...]],
                     dim=0)
                self.rnn_states[recurrent_submodule_name]['cell'][idx] = torch.cat(
                    [self.rnn_states[recurrent_submodule_name]['cell'][idx][:batch_idx,...], 
                     self.rnn_states[recurrent_submodule_name]['cell'][idx][batch_idx+1:,...]],
                     dim=0)
                
    @staticmethod
    def _extract_from_rnn_states(rnn_states_batched: dict, batch_idx: int):
        rnn_states = {k: {'hidden':[], 'cell':[]} for k in rnn_states_batched}
        for recurrent_submodule_name in rnn_states_batched:
            if rnn_states[recurrent_submodule_name] is None: continue
            for idx in range(len(rnn_states_batched[recurrent_submodule_name]['hidden'])):
                rnn_states[recurrent_submodule_name]['hidden'].append( rnn_states_batched[recurrent_submodule_name]['hidden'][idx][batch_idx,...].unsqueeze(0))
                rnn_states[recurrent_submodule_name]['cell'].append( rnn_states_batched[recurrent_submodule_name]['cell'][idx][batch_idx,...].unsqueeze(0))
        return rnn_states

    def _post_process(self, prediction):
        for k, v in prediction.items():
            if isinstance(v, dict):
                for vk in v:
                    hs, cs = v[vk]['hidden'], v[vk]['cell']
                    for idx in range(len(hs)):
                        hs[idx] = hs[idx].detach().cpu()
                        cs[idx] = cs[idx].detach().cpu()
                    prediction[k][vk] = {'hidden': hs, 'cell': cs}
            else:
                prediction[k] = v.detach().cpu()
        
        return prediction

    @staticmethod
    def _extract_from_prediction(prediction: dict, batch_idx: int):
        out_pred = dict()
        for k, v in prediction.items():
            if isinstance(v, dict):
                continue
            out_pred[k] = v[batch_idx,...].unsqueeze(0)
        return out_pred

    def handle_experience(self, s, a, r, succ_s, done):
        '''
        Note: the batch size may differ from the nbr_actor as soon as some
        actors' episodes end before the others...

        :param s: numpy tensor of states of shape batch x state_shape.
        :param a: numpy tensor of actions of shape batch x action_shape.
        :param r: numpy tensor of rewards of shape batch x reward_shape.
        :param succ_s: numpy tensor of successive states of shape batch x state_shape.
        :param done: list of boolean (batch=nbr_actor) x state_shape.
        '''
        if not self.training: return
        
        state, r, succ_state, non_terminal = self.preprocess_environment_signals(s, r, succ_s, done)
        a = torch.from_numpy(a)
        # batch x ...

        # We assume that this function has been called directly after take_action:
        # therefore the current prediction correspond to this experience.

        batch_index = -1
        done_actors_among_notdone = []
        for actor_index in range(self.nbr_actor):
            # If this actor is already done with its episode:  
            if self.previously_done_actors[actor_index]:
                continue
            # Otherwise, there is bookkeeping to do:
            batch_index +=1
            
            # Bookkeeping of the actors whose episode just ended:
            if done[actor_index] and not(self.previously_done_actors[actor_index]):
                done_actors_among_notdone.append(batch_index)
            
            actor_s = state[batch_index,...].unsqueeze(0)
            actor_a = a[batch_index,...].unsqueeze(0)
            actor_r = r[batch_index,...].unsqueeze(0)
            actor_succ_s = succ_state[batch_index,...].unsqueeze(0)
            # Watch out for the miss-match: done is a list of nbr_actor booleans,
            # which is not sync with batch_index, purposefully...
            actor_non_terminal = non_terminal[actor_index,...].unsqueeze(0)

            actor_prediction = I2AAgent._extract_from_prediction(self.current_prediction, batch_index)
            
            rnd_dict = dict()
            if self.use_rnd:
                int_reward, target_int_f = self.algorithm.compute_intrinsic_reward(actor_succ_s)
                rnd_dict = {'int_r':int_reward, 'target_int_f':target_int_f}

            if self.recurrent:
                actor_prediction['rnn_states'] = I2AAgent._extract_from_rnn_states(self.current_prediction['rnn_states'],batch_index)
                actor_prediction['next_rnn_states'] = I2AAgent._extract_from_rnn_states(self.current_prediction['next_rnn_states'],batch_index)
            
            self.update_experience_storage( storage_idx=actor_index, 
                                            state=actor_s,
                                            action=actor_a,
                                            reward=actor_r,
                                            succ_s=actor_succ_s,
                                            notdone=actor_non_terminal,
                                            current_prediction=actor_prediction,
                                            rnd_dict=rnd_dict)
            
            self.previously_done_actors[actor_index] = done[actor_index]
            self.handled_experiences +=1

        if len(done_actors_among_notdone):
            # Regularization of the agents' actors:
            done_actors_among_notdone.sort(reverse=True)
            for batch_idx in done_actors_among_notdone:
                self.update_actors(batch_idx=batch_idx)
        
        if self.training:
            if (self.handled_experiences % self.algorithm.environment_model_update_horizon*self.nbr_actor) == 0:
                self.algorithm.train_environment_model()
            if (self.handled_experiences % self.algorithm.distill_policy_update_horizon*self.nbr_actor) == 0:
                self.algorithm.train_distill_policy()
            if (self.handled_experiences % self.algorithm.model_update_horizon*self.nbr_actor) == 0:
                self.algorithm.train_i2a_model()
                if self.save_path is not None: torch.save(self, self.save_path)

    def preprocess_environment_signals(self, state, reward, succ_state, done):
        '''
        Preprocesses the various environment signals collected by the agent,
        and given as :params: to this function, to be used by the agent's learning algorithm.
        :returns: preprocessed state, reward, successor_state and done input paramters
        '''
        non_terminal = torch.from_numpy(1 - np.array(done)).type(torch.FloatTensor)
        s = self.preprocess_function(state, use_cuda=False)
        succ_s = self.preprocess_function(succ_state, use_cuda=False)
        if succ_s.dtype != torch.float32 or s.dtype != torch.float32: raise 
        if isinstance(reward, np.ndarray): r = torch.from_numpy(reward).type(torch.FloatTensor)
        else: r = torch.ones(1).type(torch.FloatTensor)*reward
        return s, r, succ_s, non_terminal

    def update_experience_storage(self, storage_idx: int,
                                   state: torch.Tensor, action: torch.Tensor,
                                   reward: torch.Tensor, succ_s: torch.Tensor,
                                   notdone: torch.Tensor, current_prediction: Dict[str, object],
                                   rnd_dict: Dict[str, object]):
        '''
        Adds the already preprocessed state signals to the different storage buffers
        used by the underlying I2AAlgorithm. In the current version a separate storage
        is being used for the environment model, distill policy and the I2A model.
        It may be possible to share storages in the future to reduce memory requirements
        '''
        environment_model_relevant_info = {'s': state,
                                           'a': current_prediction['a'],
                                           'r': reward,
                                           'succ_s': succ_s,
                                           'non_terminal': notdone}
        if self.use_rnd:
            int_r = rnd_dict['int_r']*self.algorithm.kwargs['rnd_loss_int_ratio']
            # Normalization of intrinsic reward:
            int_r = int_r+self.algorithm.model_training_algorithm.int_reward_std+1e-8
            environment_model_relevant_info['r'] = environment_model_relevant_info['r'] + int_r
        #environment_model_relevant_info.update(rnd_dict)
        self.algorithm.environment_model_storages[storage_idx].add(environment_model_relevant_info)

        distill_policy_relevant_info = {'s': state,
                                        'a': current_prediction['a']}
        distill_policy_relevant_info.update(current_prediction)
        if self.use_rnd:
            distill_policy_relevant_info.pop('int_v')
        self.algorithm.distill_policy_storages[storage_idx].add(distill_policy_relevant_info)

        model_relevant_info = {'s': state,
                               'r': reward,
                               'succ_s': succ_s,
                               'non_terminal': notdone}
        model_relevant_info.update(current_prediction)
        model_relevant_info.update(rnd_dict)
        self.algorithm.model_training_algorithm.storages[storage_idx].add(model_relevant_info)
        
        '''
        if self.training and self.handled_experiences % self.algorithm.kwargs['horizon'] == 0:
            next_prediction = self._post_process(self._make_prediction(succ_s))
            self.algorithm.model_training_algorithm.storages[storage_idx].add(next_prediction)
        '''

    def take_action(self, state: np.ndarray) -> np.ndarray:
        preprocessed_state = self.preprocess_function(state, use_cuda=self.use_cuda)
        # The I2A model will take care of its own rnn state:
        self.current_prediction = self.algorithm.take_action(preprocessed_state)
        self.current_prediction = self._post_process(self.current_prediction)
        return self.current_prediction['a'].numpy()

    def _make_prediction(self, preprocessed_state: torch.Tensor) -> Dict[str, object]:
        prediction = self.algorithm.take_action(preprocessed_state)
        return prediction

    def clone(self, training: bool = None):
        return NotImplementedError(f'Clone function for  {self.__class__} \
                                     algorithm not yet supported')


def build_environment_model(task, kwargs: Dict[str, object]) -> EnvironmentModel:
    '''
    Creates an environment model which given an observation and an action will
    return a successor observation and a reward associated to the approximated model
    transition function. This environment model will be used to 'imagine' rollouts
    as part of the I2A algorithm. Refer to original paper Figure (1)

    :returns: torch.nn.Module which approximates a transition probability function
    '''
    if kwargs['environment_model_arch'] == 'Sokoban':
        conv_dim = kwargs['environment_model_channels'][0]
        model = EnvironmentModel(observation_shape=kwargs['preprocessed_observation_shape'],
                                 num_actions=task.action_dim,
                                 reward_size=kwargs['reward_size'],
                                 conv_dim=conv_dim,
                                 use_cuda=kwargs['use_cuda'])
    elif kwargs['environment_model_arch'] == 'MLP':
        enc_input_dim = kwargs['latent_emb_nbr_variables']+task.action_dim
        encoder = FCBody(enc_input_dim, hidden_units=kwargs['environment_model_enc_nbr_hidden_units'], gate=F.leaky_relu)
        dec_hidden_units = kwargs['environment_model_dec_nbr_hidden_units']+(kwargs['latent_emb_nbr_variables'],)
        dec_input_dim = kwargs['environment_model_enc_nbr_hidden_units'][-1]
        decoder = FCBody(dec_input_dim, hidden_units=dec_hidden_units, gate=F.leaky_relu)
        model = AutoEncoderEnvironmentModel(encoder=encoder,
                                            decoder=decoder,
                                            observation_shape=[kwargs['latent_emb_nbr_variables']],
                                            num_actions=task.action_dim,
                                            reward_size=kwargs['reward_size'],
                                            use_cuda=kwargs['use_cuda'])
    
    else:
        raise NotImplementedError('Environment model: only the CNN architecture has been implemented.')

    return model


def build_model_free_network(kwargs: Dict[str, object]) -> nn.Module:
    '''
    Creates a neural network architecture to be used as the model
    free pass of the I2A model architecture. This module receives
    an observation as input and outputs a latent feature vector.
    Refer to original paper Figure (1).
    :returns: torch.nn.Module used as part of I2A's policy model
    '''
    input_shape = kwargs['preprocessed_observation_shape']
    if kwargs['use_latent_embedding']: input_shape = [kwargs['latent_emb_nbr_variables']]
    model = choose_architecture(architecture=kwargs['model_free_network_arch'],
                                input_shape=input_shape,
                                hidden_units_list=kwargs['model_free_network_nbr_hidden_units'],
                                feature_dim=kwargs['model_free_network_feature_dim'],
                                nbr_channels_list=kwargs['model_free_network_channels'],
                                kernels=kwargs['model_free_network_kernels'],
                                strides=kwargs['model_free_network_strides'],
                                paddings=kwargs['model_free_network_paddings'])
    return model


def build_actor_critic_head(task, input_dim, kwargs: Dict[str, object]) -> nn.Module:
    '''
    Creates a neural network architecture to be used as the
    dual head of the I2A model architecture. One head is the policy head (actor)
    and the other is the value head (critic). Refer to original paper Figure (1).
    :returns: torch.nn.Module used as part of I2A's policy model
    '''
    phi_body = choose_architecture(architecture=kwargs['achead_phi_arch'],
                                   input_shape=[input_dim],
                                   hidden_units_list=kwargs['achead_phi_nbr_hidden_units'])
    input_dim = phi_body.get_feature_shape()
    actor_body = choose_architecture(architecture=kwargs['achead_actor_arch'],
                                     input_shape=[input_dim],
                                     hidden_units_list=kwargs['achead_actor_nbr_hidden_units'])
    critic_body = choose_architecture(architecture=kwargs['achead_critic_arch'],
                                      input_shape=[input_dim],

                                      hidden_units_list=kwargs['achead_critic_nbr_hidden_units'])

    if task.action_type == 'Discrete' and task.observation_type == 'Discrete':
        model = CategoricalActorCriticNet(task.observation_shape, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=critic_body,
                                          use_intrinsic_critic=kwargs['use_random_network_distillation'])
    if task.action_type == 'Discrete' and task.observation_type == 'Continuous':
        model = CategoricalActorCriticNet(task.observation_shape, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=critic_body,
                                          use_intrinsic_critic=kwargs['use_random_network_distillation'])
    return model


def choose_model_training_algorithm(model_training_algorithm: str, kwargs: Dict[str, object]):
    '''
    The I2A architecture is mostly agnostic to which algorithm is used to compute the loss
    function which will update the I2A model's parameters. And hence, it is theoretically
    possible to use many different RL algorithms. Currently only PPO is supported as
    a 'backend' algorithm.
    :returns: regym.rl_algorithms algorithm used to compute the loss function which
              will update I2A's model parameters.
    '''
    if 'PPO' in model_training_algorithm:
        from regym.rl_algorithms.PPO import PPOAlgorithm
        PPOAlgorithm.check_mandatory_kwarg_arguments(kwargs)

        target_intr_model = None
        predict_intr_model = None
        if kwargs['use_random_network_distillation']:
            if kwargs['rnd_arch'] == 'MLP':
                target_intr_model = FCBody(task.observation_shape, hidden_units=kwargs['rnd_feature_net_fc_arch_hidden_units'], gate=F.leaky_relu)
                predict_intr_model = FCBody(task.observation_shape, hidden_units=kwargs['rnd_feature_net_fc_arch_hidden_units'], gate=F.leaky_relu)
            elif 'CNN' in kwargs['rnd_arch']:
                input_shape = kwargs['preprocessed_observation_shape']
                channels = [input_shape[0]] + kwargs['rnd_arch_channels']
                kernels = kwargs['rnd_arch_kernels']
                strides = kwargs['rnd_arch_strides']
                paddings = kwargs['rnd_arch_paddings']
                output_dim = kwargs['rnd_arch_feature_dim']
                target_intr_model = ConvolutionalBody(input_shape=input_shape,
                                                      feature_dim=output_dim,
                                                      channels=channels,
                                                      kernel_sizes=kernels,
                                                      strides=strides,
                                                      paddings=paddings)
                predict_intr_model = ConvolutionalBody(input_shape=input_shape,
                                                      feature_dim=output_dim,
                                                      channels=channels,
                                                      kernel_sizes=kernels,
                                                      strides=strides,
                                                      paddings=paddings)
            target_intr_model.share_memory()
            predict_intr_model.share_memory()

        return partial(PPOAlgorithm, target_intr_model=target_intr_model, predict_intr_model=predict_intr_model)
    raise ValueError(f"I2A agent currently only supports 'PPO' \
                      as a training algorithm. Given {model_training_algorithm}")


def build_rollout_encoder(task, kwargs: Dict[str, object]) -> nn.Module:
    input_shape = kwargs['preprocessed_observation_shape']
    if kwargs['use_latent_embedding']: input_shape = [kwargs['latent_emb_nbr_variables']]

    if kwargs['rollout_encoder_model_arch'] == 'CNN-GRU-RNN':
        feature_encoder = choose_architecture(architecture='CNN',
                                              input_shape=input_shape,
                                              hidden_units_list=None,
                                              feature_dim=kwargs['rollout_encoder_feature_dim'],
                                              nbr_channels_list=kwargs['rollout_encoder_channels'],
                                              kernels=kwargs['rollout_encoder_kernels'],
                                              strides=kwargs['rollout_encoder_strides'],
                                              paddings=kwargs['rollout_encoder_paddings'])
    elif kwargs['rollout_encoder_model_arch'] == 'MLP-GRU-RNN':
        feature_encoder = choose_architecture(architecture='MLP',
                                              input_shape=input_shape,
                                              hidden_units_list=kwargs['rollout_encoder_nbr_hidden_units'])

    rollout_feature_encoder_input_dim = feature_encoder.get_feature_shape()+kwargs['reward_size']
    rollout_feature_encoder = nn.LSTM(input_size=rollout_feature_encoder_input_dim,
                                      hidden_size=kwargs['rollout_encoder_encoder_nbr_hidden_units'],
                                      num_layers=kwargs['rollout_encoder_nbr_rnn_layers'],
                                      batch_first=False,
                                      dropout=0.0,
                                      bidirectional=False)
    
    rollout_encoder = RolloutEncoder(input_shape=input_shape,
                                     nbr_states_to_encode=min(kwargs['rollout_length'], kwargs['rollout_encoder_nbr_state_to_encode']),
                                     feature_encoder=feature_encoder,
                                     rollout_feature_encoder=rollout_feature_encoder,
                                     kwargs=kwargs)
    return rollout_encoder

class concat_aggr(object):
    def __call__(self, rollout_embeddings):
        batch_size = rollout_embeddings.size(0)
        return rollout_embeddings.view(batch_size, -1)

def build_aggregator(task):
    '''
    input dimensions to the aggregator:
    batch x imagined_rollouts_per_step x (rollout_embedding_size+nbr_action)
    returns: aggregator class/function to be used as part of a I2AModel
    '''
    aggr_fn = concat_aggr()
    return aggr_fn


def build_distill_policy(task, kwargs: Dict[str, object]) -> nn.Module:
    input_dim = task.observation_shape
    if kwargs['use_latent_embedding']: input_dim = kwargs['latent_emb_nbr_variables']

    if kwargs['distill_policy_arch'] == 'MLP':
        phi_body = FCBody(input_dim, hidden_units=kwargs['distill_policy_nbr_hidden_units'], gate=F.leaky_relu)
        phi_body_output_dim = kwargs['distill_policy_nbr_hidden_units'][-1]
    elif kwargs['distill_policy_arch'] == 'CNN':
        # Technical debt add shape of env.observation_space to environment parser
        channels = [kwargs['preprocessed_observation_shape'][0]] + kwargs['distill_policy_channels']
        phi_body = ConvolutionalBody(input_shape=kwargs['preprocessed_observation_shape'],
                                     feature_dim=kwargs['distill_policy_feature_dim'],
                                     channels=channels,
                                     kernel_sizes=kwargs['distill_policy_kernels'],
                                     strides=kwargs['distill_policy_strides'],
                                     paddings=kwargs['distill_policy_paddings'])
        phi_body_output_dim = kwargs['distill_policy_feature_dim']
    else:
        phi_body = None 
        phi_body_output_dim = input_dim

    if kwargs['distill_policy_head_arch'] == 'LSTM-RNN':
        actor_body = LSTMBody(phi_body_output_dim, hidden_units=kwargs['distill_policy_head_nbr_hidden_units'], gate=F.leaky_relu)
    elif kwargs['distill_policy_head_arch'] == 'MLP':
        actor_body = FCBody(phi_body_output_dim, hidden_units=kwargs['distill_policy_head_nbr_hidden_units'], gate=F.leaky_relu)

    # TECHNICAL DEBT: The distill policy is only an actor, we shold not be using
    # an actor critic net as we never make use of the critic.
    if task.action_type == 'Discrete' and task.observation_type == 'Discrete':
        model = CategoricalActorCriticNet(input_dim, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=None)
    if task.action_type == 'Discrete' and task.observation_type == 'Continuous':
        model = CategoricalActorCriticNet(input_dim, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=None)
    if kwargs['use_cuda']: model = model.cuda()

    return model


def build_latent_encoder(task, kwargs: Dict[str, object]) -> nn.Module:
    observation_shape = kwargs['preprocessed_observation_shape']
    channels = [observation_shape[0]] + kwargs['latent_encoder_channels']
    latent_encoder = ConvolutionalBody(input_shape=observation_shape,
                                 feature_dim=kwargs['latent_encoder_feature_dim'],
                                 channels=channels,
                                 kernel_sizes=kwargs['latent_encoder_kernels'],
                                 strides=kwargs['latent_encoder_strides'],
                                 paddings=kwargs['latent_encoder_paddings'])

    '''
    latent_encoder = BroadcastBetaVAE(encoder=encoder,
                                      decoder=decoder,
                                      observation_shape=observation_shape)
    '''
    return latent_encoder


def build_I2A_Agent(task, config: Dict[str, object], agent_name: str) -> I2AAgent:
    '''
    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param config: Dictionary whose entries contain hyperparameters for the A2C agents:
        - 'rollout_length': Number of steps to take in every imagined rollout (length of imagined rollouts)
        - 'imagined_rollouts_per_step': Number of imagined trajectories to compute at each inference of the I2A model.
                                        If None (default), it will be replaced by the size of the action space of the task.
        - 'environment_update_horizon': (0, infinity) Number of timesteps that will elapse in between optimization calls.
        - 'policies_update_horizon': (0, infinity) Number of timesteps that will elapse in between optimization calls.
        - 'environment_model_learning_rate':
        - 'environment_model_adam_eps':
        - 'policies_learning_rate':
        - 'policies_adam_eps':
        - 'use_cuda': Whether or not to use CUDA to speed up training

        BIG TODO: Add all possible hyper parameters defined in our I2A integration test
    '''
    # Given the dependance on another training algorithm to train the model,
    # the horizon value used by this training algorithm ought to be set by
    # the hyperparamet 'model_update_horizon'...
    config['horizon'] = config['model_update_horizon']
    if config['imagined_rollouts_per_step'] is None: 
        config['imagined_rollouts_per_step'] = task.action_dim

    # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
    preprocess_function = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
    config['preprocessed_observation_shape'] = [task.observation_shape[-1], config['observation_resize_dim'], config['observation_resize_dim']]
    if 'nbr_frame_stacking' in config:
        config['preprocessed_observation_shape'][0] *= config['nbr_frame_stacking']
            
    environment_model = build_environment_model(task, config)

    distill_policy = build_distill_policy(task, config)

    imagination_core    = ImaginationCore(distill_policy=distill_policy, environment_model=environment_model)

    rollout_encoder     = build_rollout_encoder(task, config)

    model_training_algorithm_class = choose_model_training_algorithm(config['model_training_algorithm'], config)
    aggregator = build_aggregator(task)
    model_free_network = build_model_free_network(config)

    actor_critic_input_dim = model_free_network.get_feature_shape()+(config['rollout_encoder_embedding_size']+task.action_dim)*config['imagined_rollouts_per_step']
    actor_critic_head = build_actor_critic_head(task, input_dim=actor_critic_input_dim, kwargs=config)

    latent_encoder = None
    if config['use_latent_embedding']:
        latent_encoder = build_latent_encoder(task,config)

    recurrent_submodule_names = [key for key, value in config.items() if isinstance(value, str) and 'RNN' in value]
    i2a_model = I2AModel(actor_critic_head=actor_critic_head,
                         model_free_network=model_free_network,
                         aggregator=aggregator,
                         rollout_encoder=rollout_encoder,
                         imagination_core=imagination_core,
                         imagined_rollouts_per_step=config['imagined_rollouts_per_step'],
                         rollout_length=config['rollout_length'],
                         latent_encoder=latent_encoder,
                         rnn_keys=recurrent_submodule_names,
                         kwargs=config)

    algorithm = I2AAlgorithm(model_training_algorithm_init_function=model_training_algorithm_class,
                             i2a_model=i2a_model,
                             environment_model=environment_model,
                             distill_policy=distill_policy,
                             kwargs=config,
                             latent_encoder=latent_encoder)

    return I2AAgent(algorithm=algorithm, name=agent_name,
                    preprocess_function=preprocess_function,
                    rnn_keys=recurrent_submodule_names,
                    use_cuda=config['use_cuda'])
