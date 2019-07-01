import gym.spaces
import torch
import torch.nn as nn
import torch.nn.functional as F

from regym.rl_algorithms.networks import CNNPreprocessFunction, PreprocessFunction
from regym.rl_algorithms.I2A import I2AAlgorithm, ImaginationCore
from regym.rl_algorithms.networks import CategoricalActorCriticNet, FCBody, LSTMBody, ConvolutionalBody
from regym.rl_algorithms.networks.utils import output_size_for_model


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

    def handle_experience(self, s, a, r, succ_s, done=False):
        '''
        Info that will be needed to be stored at this point:
          - state
          - action taken
          - log probabilities of all actions from actor (I2A)
          - log probabilities of distilled (rollout policy)
          - value estimation for the state
        Put in storage all
        '''
        if not self.training: return
        self.handled_experiences += 1

        non_terminal = torch.ones(1)*(1 - int(done))
        # Add current_prediction to storage (current_prediction needs to be computed in take_action)
        self.algorithm.storage.add({'r': r, 'non_terminal': non_terminal, 's':  s})
        if (self.handled_experiences % self.algorithm.environment_update_horizon) == 0:
            self.algorithm.train_environment_model()
        if (self.handled_experiences % self.algorithm.policies_update_horizon) == 0:
            self.algorithm.train_policies()

    def take_action(self, state):
        '''
        TODO: call self.algorithm.actor_critic() to get:
                  - action taken
                  - log probabilities of all actions from actor (I2A)
                  - log probabilities of distilled (rollout policy)
                  - value estimation for the state from critic
        Put all in self.current_prediction
        '''
        processed_state = self.preprocess_function(state, use_cuda=self.algorithm.use_cuda)
        prediction = self.algorithm.take_action(processed_state)
        return prediction['a'].item()

    def clone(self, training=None):
        pass


def choose_model_free_network(task, kwargs):
    model = choose_architecture(task, task.observation_dim, architecture=kwargs['model_free_network_arch'],
                                feature_dim=kwargs['model_free_network_feature_dim'],
                                nbr_channels_list=kwargs['model_free_network_channels'],
                                kernels=kwargs['model_free_network_kernels'],
                                strides=kwargs['model_free_network_strides'],
                                paddings=kwargs['model_free_network_paddings'])
    return model


def choose_architecture(task, input_dim, architecture, hidden_units_list=None,
                        feature_dim=None, nbr_channels_list=None, kernels=None, strides=None, paddings=None):
    if architecture == 'RNN':
        return LSTMBody(input_dim, hidden_units=hidden_units_list, gate=F.leaky_relu)
    if architecture == 'MLP':
        return FCBody(input_dim, hidden_units=hidden_units_list, gate=F.leaky_relu)
    if architecture == 'CNN':
        channels = [task.env.observation_space.shape[-1]] + nbr_channels_list
        phi_body = ConvolutionalBody(input_shapes=task.env.observation_space.shape,
                                     feature_dim=feature_dim,
                                     channels=channels,
                                     kernel_sizes=kernels,
                                     strides=strides,
                                     paddings=paddings)
        return phi_body


def choose_actor_critic_head(task, input_dim, kwargs):
    actor_body = choose_architecture(task, input_dim, kwargs['actor_critic_head_actor_arch'],
                                     kwargs['actor_critic_head_actor_nbr_hidden_units'])
    critic_body = choose_architecture(task, input_dim, kwargs['actor_critic_head_critic_arch'],
                                      kwargs['actor_critic_head_critic_nbr_hidden_units'])

    if task.action_type == 'Discrete' and task.observation_type == 'Discrete':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=None,
                                          actor_body=actor_body, critic_body=critic_body)
    if task.action_type == 'Discrete' and task.observation_type == 'Continuous':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=None,
                                          actor_body=actor_body, critic_body=critic_body)
    return model


def choose_rollout_encoder(task):
    return None


def choose_aggregator(task):
    # lambda x: torch.cat(x) Simple aggretator which only concatenates
    return None


def choose_distill_policy(task, kwargs):
    input_dim = task.observation_dim
    if kwargs['distill_policy_arch'] == 'MLP':
        phi_body = FCBody(input_dim, hidden_units=kwargs['distill_policy_nbr_hidden_units'], gate=F.leaky_relu)
        input_dim = kwargs['distill_policy_nbr_hidden_units'][-1]
    elif kwargs['distill_policy_arch'] == 'CNN':
        # Technical debt add shape of env.observation_space to environment parser
        channels = [task.env.observation_space.shape[-1]] + kwargs['distill_policy_channels']
        phi_body = ConvolutionalBody(input_shapes=task.env.observation_space.shape,
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
    preprocess_function = CNNPreprocessFunction if 'CNNPreprocessFunction' in config['preprocess_function'] else PreprocessFunction
    environment_model = None

    distill_policy_model = choose_distill_policy(task, config)

    imagination_core    = ImaginationCore(distill_policy=distill_policy_model, environment_model=environment_model)

    rollout_encoder     = choose_rollout_encoder(task)
    rollout_encoder_hidden_dim = None

    aggregator = choose_aggregator(task)
    model_free_network = choose_model_free_network(task, config)

    # TODO once rollout encoder is in place (rollout_encoder_hidden_dim * config['imagined_rollouts_per_step'])
    actor_critic_input_dim = output_size_for_model(model_free_network,
                                                   input_shape=task.env.observation_space.sample().transpose((2, 0, 1)).shape)
    actor_critic_head = choose_actor_critic_head(task, input_dim=actor_critic_input_dim, kwargs=config)

    algorithm = I2AAlgorithm(imagination_core=imagination_core,
                             model_free_network=model_free_network,
                             rollout_encoder=rollout_encoder,
                             aggregator=aggregator,
                             actor_critic_head=actor_critic_head,
                             rollout_length=config['rollout_length'],
                             imagined_rollouts_per_step=config['imagined_rollouts_per_step'],
                             policies_update_horizon=config['policies_update_horizon'],
                             environment_update_horizon=config['environment_update_horizon'],
                             environment_model_learning_rate=config['environment_model_learning_rate'],
                             environment_model_adam_eps=config['environment_model_adam_eps'],
                             policies_adam_learning_rate=config['policies_learning_rate'],
                             policies_adam_eps=config['policies_adam_eps'],
                             use_cuda=config['use_cuda'])
    return I2AAgent(algorithm=algorithm, name=agent_name, action_dim=task.action_dim,
                    preprocess_function=preprocess_function)
