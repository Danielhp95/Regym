import gym.spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from regym.rl_algorithms.PPO import PPOAlgorithm
from regym.rl_algorithms.networks import ResizeCNNPreprocessFunction
from regym.rl_algorithms.I2A import I2AAlgorithm, ImaginationCore, EnvironmentModel, RolloutEncoder
from regym.rl_algorithms.networks import CategoricalActorCriticNet, FCBody, LSTMBody, ConvolutionalBody, choose_architecture

class I2AModel(nn.Module):
  def __init__(self, actor_critic_head, model_free_network, aggregator, rollout_encoder, imagination_core):
    super(I2AModel, self).__init__()

    self.actor_critic_head = actor_critic_head
    self.model_free_network = model_free_network
    self.aggregator = aggregator
    self.rollout_encoder = rollout_encoder
    self.imagination_core = imagination_core

  def __forward__(self, state, imagined_rollouts_per_step, rollout_length):
    '''
    :param state: preprocessed observation/state as a PyTorch Tensor
                  of dimensions batch_size=1 x input_shape
    :param imagined_rollouts_per_step: number of rollouts to 
                  imagine at each inference state.
    :param rollout_length: nbr of steps per rollout.
    '''
    rollout_embeddings = []
    for i in range(imagined_rollouts_per_step):
        # 1. Imagine state and reward for self.imagined_rollouts_per_step times
        rollout_states, rollout_rewards = self.imagination_core.imagine_rollout(state, rollout_length)
        # dimensions: rollout_length x batch x input_shape / reward-size
        # 2. encode them with RolloutEncoder and use aggregator to concatenate them together into imagination code
        rollout_embedding = self.rollout_encoder(rollout_states, rollout_rewards)
        # dimensions: batch x rollout_encoder_embedding_size
        rollout_embeddings.append(rollout_embedding.unsqueeze(1))
    rollout_embeddings = torch.cat(rollout_embeddings, dim=1)
    # dimensions: batch x imagined_rollouts_per_step x rollout_encoder_embedding_size
    imagination_code = self.aggregator(rollout_embeddings)
    # dimensions: batch x imagined_rollouts_per_step*rollout_encoder_embedding_size
    # 3. model free pass
    features = self.model_free_network(state)
    # dimensions: batch x model_free_feature_dim
    # 4. concatenate model free pass and imagination code
    imagination_code_features = torch.cat([imagination_code, features], dim=1)
    # 5. Final fully connected layer which turns into action.
    prediction = self.actor_critic_head(imagination_code_features)
    return prediction


class I2AAgent():

    def __init__(self, name, algorithm, action_dim, preprocess_function):
        '''
        :param name: String identifier for the agent
        :param samples_before_update: Number of actions the agent will take before updating
        :param algorithm: Reinforcement Learning algorithm used to update the agent's policy.
                          Contains the agent's policy, represented as a neural network.
        :param preprocess_function: Function which preprocesses the state before
                                    being handed to the algorithm
        '''
        self.name = name
        self.algorithm = algorithm
        self.training = True
        self.preprocess_function = preprocess_function

        self.action_dim = action_dim
        self.handled_experiences = 0
        # Current_prediction stores information
        # from the last action that was taken
        self.current_prediction = None

    def handle_experience(self, s, a, r, succ_s, done=False):
        if not self.training: return
        self.handled_experiences += 1

        state, reward, succ_s, non_terminal = self.preprocess_environment_signals(s, r, succ_s, done)
        self.update_experience_storages(state, a, reward, succ_s, non_terminal,
                                        self.current_prediction)

        if (self.handled_experiences % self.algorithm.environment_update_horizon) == 0:
            self.algorithm.train_environment_model()
        if (self.handled_experiences % self.algorithm.policies_update_horizon) == 0:
            self.algorithm.train_policies()

    def preprocess_environment_signals(self, state, reward, succ_s, done):
        state = self.preprocess_function(state, use_cuda=self.algorithm.use_cuda)
        succ_s = self.preprocess_function(succ_s, use_cuda=self.algorithm.use_cuda)
        reward = torch.ones(1)*reward
        non_terminal = torch.ones(1)*(1 - int(done))
        return state, reward, succ_s, non_terminal

    def update_experience_storages(self, state, action, reward, succ_s, done, current_prediction):
        environment_model_relevant_info = {'s': state,
                                           'a': current_prediction['a'],
                                           'r': reward,
                                           'succ_s': succ_s,
                                           'non_terminal': done}
        self.algorithm.environment_model_storage.add(environment_model_relevant_info)

        policies_relevant_info = {'r': reward, 's': state,
                                  'non_terminal': done}
        self.algorithm.policies_storage.add(current_prediction)
        self.algorithm.policies_storage.add(policies_relevant_info)

    def take_action(self, state):
        processed_state = self.preprocess_function(state, use_cuda=self.algorithm.use_cuda)
        self.current_prediction = self.algorithm.take_action(processed_state)
        return self.current_prediction['a'].item()

    def clone(self, training=None):
        pass


def build_environment_model(task, kwargs):
    if kwargs['environment_model_arch'] == 'CNN':
        conv_dim = kwargs['environment_model_channels'][0]
        model = EnvironmentModel(observation_shape=kwargs['preprocessed_observation_shape'],
                                 num_actions=task.action_dim,
                                 reward_size=kwargs['reward_size'],
                                 conv_dim=conv_dim)
    else:
        raise NotImplementedError('Environment model: only the CNN architecture has been implemented.')

    return model


def build_model_free_network(kwargs):
    model = choose_architecture(architecture=kwargs['model_free_network_arch'],
                                input_dim=kwargs['observation_resize_dim'],
                                input_shape=kwargs['preprocessed_observation_shape'],
                                feature_dim=kwargs['model_free_network_feature_dim'],
                                nbr_channels_list=kwargs['model_free_network_channels'],
                                kernels=kwargs['model_free_network_kernels'],
                                strides=kwargs['model_free_network_strides'],
                                paddings=kwargs['model_free_network_paddings'])
    return model


def build_actor_critic_head(task, input_dim, kwargs):
    actor_body = choose_architecture(architecture=kwargs['actor_critic_head_actor_arch'],
                                     input_dim=input_dim,
                                     hidden_units_list=kwargs['actor_critic_head_actor_nbr_hidden_units'])
    critic_body = choose_architecture(architecture=kwargs['actor_critic_head_critic_arch'],
                                      input_dim=input_dim,

                                      hidden_units_list=kwargs['actor_critic_head_critic_nbr_hidden_units'])

    if task.action_type == 'Discrete' and task.observation_type == 'Discrete':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=None,
                                          actor_body=actor_body, critic_body=critic_body)
    if task.action_type == 'Discrete' and task.observation_type == 'Continuous':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=None,
                                          actor_body=actor_body, critic_body=critic_body)
    return model


def choose_model_training_algorithm(model_training_algorithm: str, kwargs: dict):
    if 'PPO' in model_training_algorithm:
        from regym.rl_algorithms.PPO import PPOAlgorithm
        PPOAlgorithm.check_mandatory_kwarg_arguments(kwargs)
        return PPOAlgorithm
    raise ValueError(f"I2A agent currently only supports 'PPO' \
                      as a training algorithm. Given {model_training_algorithm}")


def build_rollout_encoder(task, kwargs):
    feature_encoder = choose_architecture(architecture='CNN',
                                          input_shape=kwargs['preprocessed_observation_shape'],
                                          hidden_units_list=None,
                                          feature_dim=kwargs['rollout_encoder_feature_dim'],
                                          nbr_channels_list=kwargs['rollout_encoder_channels'],
                                          kernels=kwargs['rollout_encoder_kernels'],
                                          strides=kwargs['rollout_encoder_strides'],
                                          paddings=kwargs['rollout_encoder_paddings'])
    rollout_feature_encoder_input_dim = feature_encoder.get_feature_size()+kwargs['reward_size']
    rollout_feature_encoder = choose_architecture(architecture='RNN',
                                                  input_dim=rollout_feature_encoder_input_dim,
                                                  hidden_units_list=kwargs['rollout_encoder_nbr_hidden_units'])

    rollout_encoder = RolloutEncoder(input_shape=kwargs['preprocessed_observation_shape'], feature_encoder=feature_encoder, rollout_feature_encoder=rollout_feature_encoder, kwargs=kwargs)
    return rollout_encoder


def build_aggregator(task):
    # input to the aggregator: dimensions: batch x imagined_rollouts_per_step x rollout_embedding_size
    class concat_aggr(object):
        def __call__(self, rollout_embeddings):
            batch_size = rollout_embeddings.size(0)
            return rollout_embeddings.view(batch_size, -1)
    aggr_fn = concat_aggr()
    return aggr_fn


def build_distill_policy(task, kwargs):
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
    #  an actor critic net as we never make use of the critic.
    if task.action_type == 'Discrete' and task.observation_type == 'Discrete':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=None)
    if task.action_type == 'Discrete' and task.observation_type == 'Continuous':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body, critic_body=None)
    return model


def build_I2A_Agent(task, config, agent_name):
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
    '''
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

    # TODO once rollout encoder is in place (rollout_encoder_hidden_dim * config['imagined_rollouts_per_step'])
    actor_critic_input_dim = config['model_free_network_feature_dim']+config['rollout_encoder_embedding_size']*config['imagined_rollouts_per_step']
    actor_critic_head = build_actor_critic_head(task, input_dim=actor_critic_input_dim, kwargs=config)

    i2a_model = I2AModel(actor_critic_head=actor_critic_head, 
                         model_free_network=model_free_network, 
                         aggregator=aggregator, 
                         rollout_encoder=rollout_encoder, 
                         imagination_core=imagination_core)
    algorithm = I2AAlgorithm(model_training_algorithm_init_function=model_training_algorithm_class,
                             i2a_model=i2a_model,
                             environment_model=environment_model,
                             distill_policy=distill_policy,
                             kwargs=config)
    return I2AAgent(algorithm=algorithm, name=agent_name, action_dim=task.action_dim,
                    preprocess_function=preprocess_function)
