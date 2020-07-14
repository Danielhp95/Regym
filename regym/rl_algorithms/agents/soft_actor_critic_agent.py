from typing import Dict, Any, List, Callable
from copy import deepcopy
import random

import gym
import torch as T
import torch.nn as nn
import numpy as np

from regym.environments import EnvType
from regym.rl_algorithms.agents import Agent
from regym.rl_algorithms.replay_buffers import ReplayBuffer, EXP
from regym.rl_algorithms.SAC import SoftActorCriticAlgorithm

from regym.networks.heads import CategoricalDuelingDQNet, CategoricalHead
from regym.networks.bodies import FCBody
from regym.networks.preprocessing import turn_into_single_element_batch


class SoftActorCriticAgent(Agent):

    def __init__(self, name: str,
                 action_space: gym.Space,
                 update_after: int,
                 update_every: int,
                 algorithm: SoftActorCriticAlgorithm,
                 state_preprocess_fn: Callable = turn_into_single_element_batch):
        super().__init__(name=name,
                         requires_environment_model=False,  # Model free!
                         multi_action_requires_server=False)
        self.action_space = action_space
        self.update_after = update_after
        self.update_every = update_every

        self.state_preprocess_fn = state_preprocess_fn

        self.algorithm = algorithm
        self.handle_experiences_since_last_update = 0

    def handle_experience(self, o, a, r, succ_o, done=False):
        super().handle_experience(o, a, r, succ_o, done)
        self.handle_experiences_since_last_update += 1

        o_prime, a_prime, r_prime, succ_o_prime = \
                self.process_environment_signals(o, a, r, succ_o)

        experience = EXP(o_prime, a_prime, succ_o_prime, r_prime, done)
        self.algorithm.replay_buffer.push(experience)

        if (self.handled_experiences > self.update_after and
                self.handle_experiences_since_last_update >= self.update_every):
            self.handle_experiences_since_last_update = 0
            self.algorithm.update()

    def model_free_take_action(self, obs, legal_actions: List[int], multi_action: bool = False):
        if self.handled_experiences < self.update_after and self.training:
            action = self.random_action(legal_actions)
        else:
            processed_o = self.state_preprocess_fn(obs)
            prediction = self.algorithm.pi_actor(processed_o,
                                                 legal_actions=legal_actions)
            if not multi_action:  # Action is a single integer
                action = np.int(prediction['a'])
            if multi_action:  # Action comes from a vector env, one action per environment
                action = prediction['a'].view(1, -1).squeeze(0).numpy()
        return action

    def random_action(self, legal_actions):
        if legal_actions: return random.choice(legal_actions)
        else: return self.action_space.sample()
        return action

    def process_environment_signals(self, o, a, r: float, succ_o):
        processed_o = self.state_preprocess_fn(o)
        processed_r = T.Tensor([r]).float()
        # TODO: processed_a Tensor is of size [action_dim],
        # shouldn't it be [1, action_dim]?
        processed_a = T.from_numpy(a).long() if isinstance(a, np.ndarray) else T.LongTensor([a])
        processed_succ_o = self.state_preprocess_fn(succ_o)
        return processed_o, processed_a, processed_r, processed_succ_o

    def clone(self, training=None):
        return NotImplementedError('Cloning not supported for SAC')


def create_critic(task, config):
    body = FCBody(task.observation_dim, hidden_units=(32, 16))  # TODO: remove magic number
    model = CategoricalDuelingDQNet(body=body,
                                    action_dim=task.action_dim)
    return model


def create_actor(task, config):
    body = FCBody(task.observation_dim, hidden_units=(32, 16))  # TODO: remove magic number
    head = CategoricalHead(body=body,
                           input_dim=body.feature_dim,
                           output_dim=task.action_dim)
    return head


def build_SAC_Agent(task: 'Task', config: Dict[str, Any],
                    agent_name: str='SAC') -> SoftActorCriticAgent:
    '''
    Implementation inspired by:
    https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/sac/sac.html


    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param config: Dict contain hyperparameters for the ExpertIterationAgent:

        - 'learning_rate': (float: [0, 1]) Learning rate for ALL neural networks (TODO: which ones)
        - 'memory_size': (int: [1, inf]) Maximum length of replay buffer.
        - 'gamma': (float: [0, 1]) MDP Discount factor. (Always between 0 and 1.)
        - 'tau': (float: [0, 1]) Interpolation factor in polyak averaging for target
                    networks. Target networks are updated towards main networks
                    according to:

                    .. math:: \\theta_{\\text{targ}} \\leftarrow
                        \\tau \\theta_{\\text{targ}} + (1-\\rho) \\theta
        - 'alpha': (float: [0, 1]) Entropy regularization coefficient.
                   (Equivalent to inverse of reward scale in the original SAC paper.)
        - 'batch_size': (int: [1, inf]) Minibatch size for computing each loss
        - 'update_after': Number of env interactions to collect before
                               starting to do gradient descent updates. Ensures replay buffer
                               is full enough for useful updates.
        - 'update_every': (int: [1, inf] Number of env interactions that should elapse
                                   between gradient descent updates.
    :returns: A SoftActorCriticAgent agent
    '''
    if task.env_type == EnvType.SINGLE_AGENT: action_space = task.env.action_space
    # Assumes all agents share same action space
    else: action_space = task.env.action_space.spaces[0]


    # If alpha is not present, follow Olivier Sigaud's intuition from:
    # https://www.youtube.com/watch?v=_nFXOZpo50U
    # Slide 19/20
    if 'alpha' in config: alpha = config['alpha']
    else: alpha = 0.98 * np.log(task.action_dim)

    # Create actor critic network, where the Critic head is a Q! Make it have 2 heads?
    pi_actor = create_actor(task, config)
    q_critic_1 = create_critic(task, config)
    q_critic_2 = create_critic(task, config)
    
    replay_buffer = ReplayBuffer(capacity=config["memory_size"])

    algorithm = SoftActorCriticAlgorithm(
            use_cuda=config['use_cuda'],
            learning_rate=config['learning_rate'],
            tau=config['tau'],
            alpha=alpha,
            gamma=config['gamma'],
            batch_size=config['batch_size'],
            memory_size=config['memory_size'],
            pi_actor = pi_actor,
            q_critic_1 = q_critic_1,
            q_critic_2 = q_critic_2,
            replay_buffer=replay_buffer)
    return SoftActorCriticAgent(
            name=agent_name,
            algorithm=algorithm,
            action_space=action_space,
            update_after=config['update_after'],
            update_every=config['update_every'])



def check_task_compatibility(task: 'Task', config: Dict[str, Any]):
    # TODO: check it only works on discrete action spaces
    if task.action_type == 'continuous':
        raise NotImplementedError('Current SoftActorCriticAgent does not support continuous action spaces')
