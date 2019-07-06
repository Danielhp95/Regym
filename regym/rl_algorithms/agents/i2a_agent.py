from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from regym.rl_algorithms.networks import ResizeCNNPreprocessFunction
from regym.rl_algorithms.I2A import I2AAlgorithm, ImaginationCore, EnvironmentModel, RolloutEncoder, I2AModel
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
        self.training = True
        self.use_cuda = use_cuda
        self.preprocess_function = preprocess_function

        self.handled_experiences = 0
        # Current_prediction stores information
        # from the last action that was taken
        self.current_prediction: Dict[str, object]

        self.recurrent = False
        self.rnn_keys = rnn_keys
        if len(self.rnn_keys):
            self.recurrent = True

    def handle_experience(self, s, a, r, succ_s, done=False):
        if not self.training: return
        self.handled_experiences += 1

        state, reward, succ_s, non_terminal = self.preprocess_environment_signals(s, r, succ_s, done)
        self.update_experience_storages(state, a, reward, succ_s, non_terminal, self.current_prediction)

        if (self.handled_experiences % self.algorithm.environment_model_update_horizon) == 0:
            self.algorithm.train_environment_model()
        if (self.handled_experiences % self.algorithm.distill_policy_update_horizon) == 0:
            self.algorithm.train_distill_policy()
        if (self.handled_experiences % self.algorithm.model_update_horizon) == 0:
            self.algorithm.train_i2a_model()

    def preprocess_environment_signals(self, state, reward, succ_s, done):
        '''
        Preprocesses the various environment signals collected by the agent,
        and given as :params: to this function, to be used by the agent's learning algorithm.
        :returns: preprocessed state, reward, successor_state and done input paramters
        '''
        state = self.preprocess_function(state, use_cuda=self.algorithm.use_cuda)
        succ_s = self.preprocess_function(succ_s, use_cuda=self.algorithm.use_cuda)
        reward = torch.Tensor([reward])
        non_terminal = torch.Tensor([1 - int(done)])
        return state, reward, succ_s, non_terminal

    def _post_process(self, prediction):
        if self.recurrent:
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
        else:
            prediction = {k: v.detach().cpu() for k, v in prediction.items()}
        return prediction

    def update_experience_storages(self, state: torch.Tensor, action: torch.Tensor,
                                   reward: torch.Tensor, succ_s: torch.Tensor,
                                   done: torch.Tensor, current_prediction: Dict[str, object]):
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
                                           'non_terminal': done}
        self.algorithm.environment_model_storage.add(environment_model_relevant_info)

        distill_policy_relevant_info = {'s': state,
                                        'a': current_prediction['a']}
        distill_policy_relevant_info.update(current_prediction)
        self.algorithm.distill_policy_storage.add(distill_policy_relevant_info)

        model_relevant_info = {'s': state,
                               'r': reward,
                               'succ_s': succ_s,
                               'non_terminal': done}
        model_relevant_info.update(current_prediction)
        self.algorithm.model_training_algorithm.storage.add(model_relevant_info)
        if self.training and self.handled_experiences % self.algorithm.kwargs['horizon'] == 0:
            next_prediction = self._post_process(self._make_prediction(succ_s))
            self.algorithm.model_training_algorithm.storage.add(next_prediction)

    def take_action(self, state):
        preprocessed_state = self.preprocess_function(state, use_cuda=self.use_cuda)
        self.current_prediction = self._make_prediction(preprocessed_state)
        self.current_prediction = self._post_process(self.current_prediction)
        return self.current_prediction['a'].item()

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
    if kwargs['environment_model_arch'] == 'CNN':
        conv_dim = kwargs['environment_model_channels'][0]
        model = EnvironmentModel(observation_shape=kwargs['preprocessed_observation_shape'],
                                 num_actions=task.action_dim,
                                 reward_size=kwargs['reward_size'],
                                 conv_dim=conv_dim,
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
    model = choose_architecture(architecture=kwargs['model_free_network_arch'],
                                input_dim=kwargs['observation_resize_dim'],
                                input_shape=kwargs['preprocessed_observation_shape'],
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
                                   input_dim=input_dim,
                                   hidden_units_list=kwargs['achead_phi_nbr_hidden_units'])
    input_dim = phi_body.get_feature_size()
    actor_body = choose_architecture(architecture=kwargs['achead_actor_arch'],
                                     input_dim=input_dim,
                                     hidden_units_list=kwargs['achead_actor_nbr_hidden_units'])
    critic_body = choose_architecture(architecture=kwargs['achead_critic_arch'],
                                      input_dim=input_dim,

                                      hidden_units_list=kwargs['achead_critic_nbr_hidden_units'])

    if task.action_type == 'Discrete' and task.observation_type == 'Discrete':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=critic_body)
    if task.action_type == 'Discrete' and task.observation_type == 'Continuous':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=critic_body)
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
        return PPOAlgorithm
    raise ValueError(f"I2A agent currently only supports 'PPO' \
                      as a training algorithm. Given {model_training_algorithm}")


def build_rollout_encoder(task, kwargs: Dict[str, object]) -> nn.Module:
    feature_encoder = choose_architecture(architecture='CNN',
                                          input_shape=kwargs['preprocessed_observation_shape'],
                                          hidden_units_list=None,
                                          feature_dim=kwargs['rollout_encoder_feature_dim'],
                                          nbr_channels_list=kwargs['rollout_encoder_channels'],
                                          kernels=kwargs['rollout_encoder_kernels'],
                                          strides=kwargs['rollout_encoder_strides'],
                                          paddings=kwargs['rollout_encoder_paddings'])
    rollout_feature_encoder_input_dim = feature_encoder.get_feature_size()+kwargs['reward_size']
    rollout_feature_encoder = nn.LSTM(input_size=rollout_feature_encoder_input_dim,
                                      hidden_size=kwargs['rollout_encoder_nbr_hidden_units'],
                                      num_layers=kwargs['rollout_encoder_nbr_rnn_layers'],
                                      batch_first=False,
                                      dropout=0.0,
                                      bidirectional=False)
    '''
    rollout_feature_encoder = choose_architecture(architecture='RNN',
                                                  input_dim=rollout_feature_encoder_input_dim,
                                                  hidden_units_list=kwargs['rollout_encoder_nbr_hidden_units'])
    '''
    rollout_encoder = RolloutEncoder(input_shape=kwargs['preprocessed_observation_shape'],
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
    batch x imagined_rollouts_per_step x rollout_embedding_size
    returns: aggregator class/function to be used as part of a I2AModel
    '''
    aggr_fn = concat_aggr()
    return aggr_fn


def build_distill_policy(task, kwargs: Dict[str, object]) -> nn.Module:
    input_dim = task.observation_dim
    if kwargs['distill_policy_arch'] == 'MLP':
        phi_body = FCBody(input_dim, hidden_units=kwargs['distill_policy_nbr_hidden_units'], gate=F.leaky_relu)
        input_dim = kwargs['distill_policy_nbr_hidden_units'][-1]
    elif kwargs['distill_policy_arch'] == 'CNN':
        # Technical debt add shape of env.observation_space to environment parser
        channels = [kwargs['preprocessed_observation_shape'][0]] + kwargs['distill_policy_channels']
        phi_body = ConvolutionalBody(input_shape=kwargs['preprocessed_observation_shape'],
                                     feature_dim=kwargs['distill_policy_feature_dim'],
                                     channels=channels,
                                     kernel_sizes=kwargs['distill_policy_kernels'],
                                     strides=kwargs['distill_policy_strides'],
                                     paddings=kwargs['distill_policy_paddings'])
        input_dim = kwargs['distill_policy_feature_dim']

    if kwargs['distill_policy_head_arch'] == 'RNN':
        actor_body = LSTMBody(input_dim, hidden_units=kwargs['distill_policy_head_nbr_hidden_units'], gate=F.leaky_relu)
    elif kwargs['distill_policy_head_arch'] == 'MLP':
        actor_body = FCBody(input_dim, hidden_units=kwargs['distill_policy_head_nbr_hidden_units'], gate=F.leaky_relu)

    # TECHNICAL DEBT: The distill policy is only an actor, we shold not be using
    # an actor critic net as we never make use of the critic.
    if task.action_type == 'Discrete' and task.observation_type == 'Discrete':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=None)
    if task.action_type == 'Discrete' and task.observation_type == 'Continuous':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=None)
    if kwargs['use_cuda']: model = model.cuda()

    return model


def build_I2A_Agent(task, config: Dict[str, object], agent_name: str) -> I2AAgent:
    '''
    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param config: Dictionary whose entries contain hyperparameters for the A2C agents:
        - 'rollout_length': Number of steps to take in every imagined rollout (length of imagined rollouts)
        - 'imagined_rollouts_per_step': Number of imagined trajectories to compute at each forward pass of the I2A (rephrase)
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

    # Assuming raw pixels input, the shape is dependant on the observation_resize_dim specified by the user:
    preprocess_function = partial(ResizeCNNPreprocessFunction, size=config['observation_resize_dim'])
    config['preprocessed_observation_shape'] = [task.env.observation_space.shape[-1], config['observation_resize_dim'], config['observation_resize_dim']]

    environment_model = build_environment_model(task, config)

    distill_policy = build_distill_policy(task, config)

    imagination_core    = ImaginationCore(distill_policy=distill_policy, environment_model=environment_model)

    rollout_encoder     = build_rollout_encoder(task, config)

    model_training_algorithm_class = choose_model_training_algorithm(config['model_training_algorithm'], config)
    aggregator = build_aggregator(task)
    model_free_network = build_model_free_network(config)

    actor_critic_input_dim = config['model_free_network_feature_dim']+config['rollout_encoder_embedding_size']*config['imagined_rollouts_per_step']
    actor_critic_head = build_actor_critic_head(task, input_dim=actor_critic_input_dim, kwargs=config)

    recurrent_submodule_names = [key for key, value in config.items() if isinstance(value, str) and 'RNN' in value]
    i2a_model = I2AModel(actor_critic_head=actor_critic_head,
                         model_free_network=model_free_network,
                         aggregator=aggregator,
                         rollout_encoder=rollout_encoder,
                         imagination_core=imagination_core,
                         imagined_rollouts_per_step=config['imagined_rollouts_per_step'],
                         rollout_length=config['rollout_length'],
                         rnn_keys=recurrent_submodule_names,
                         use_cuda=config['use_cuda'])

    algorithm = I2AAlgorithm(model_training_algorithm_init_function=model_training_algorithm_class,
                             i2a_model=i2a_model,
                             environment_model=environment_model,
                             distill_policy=distill_policy,
                             kwargs=config)
    return I2AAgent(algorithm=algorithm, name=agent_name,
                    preprocess_function=preprocess_function,
                    rnn_keys=recurrent_submodule_names,
                    use_cuda=config['use_cuda'])
