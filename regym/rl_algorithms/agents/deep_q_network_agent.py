from typing import List
import copy
import numpy as np
import random
import torch as T

import torch.nn as nn

from regym.rl_algorithms.agents import Agent
from regym.rl_algorithms.replay_buffers import EXP
from regym.networks import CategoricalDuelingDQNet, CategoricalDQNet
from regym.networks import LeakyReLU, FCBody
from regym.networks.preprocessing import turn_into_single_element_batch
from regym.rl_algorithms.DQN import DeepQNetworkAlgorithm


class DeepQNetworkAgent(Agent):
    def __init__(self, name, algorithm):
        """
        :param algorithm: algorithm class to use to optimize the network.
        """
        super(DeepQNetworkAgent, self).__init__(name=name,
                                                requires_environment_model=False)

        self.training = True
        self.kwargs = algorithm.kwargs

        self.algorithm = algorithm
        self.preprocessing_function = self.algorithm.kwargs["preprocess"]

        self.epsend = self.kwargs['epsend']
        self.epsstart = self.kwargs['epsstart']
        self.epsdecay = self.kwargs['epsdecay']
        self.nbr_steps = 0

    def getModel(self):
        return self.algorithm.model

    def handle_experience(self, s, a, r, succ_s, done=False):
        hs = self.preprocessing_function(s)
        hsucc = self.preprocessing_function(succ_s)
        r = T.ones(1)*r
        a_tensor = T.from_numpy(a) if isinstance(a, np.ndarray) else T.LongTensor([a])
        experience = EXP(hs, a_tensor, hsucc, r, done)
        self.algorithm.handle_experience(experience=experience)

        if self.training and self.algorithm.is_ready_to_train():
            self.algorithm.train(iterations=self.kwargs['nbrTrainIteration'])

    def model_free_take_action(self, state: np.ndarray, legal_actions: List[int], multi_action: bool = False):
        self.nbr_steps += 1
        self.eps = self.epsend + (self.epsstart-self.epsend) * np.exp(-1.0 * self.nbr_steps / self.epsdecay)
        action = self.select_action(model=self.algorithm.model,
                                    state=self.preprocessing_function(state),
                                    eps=self.eps,
                                    legal_actions=legal_actions,
                                    training=self.training)

        is_single_int_action = \
            lambda a: a.shape == T.Size([1, 1]) or a.shape == (1,)

        if is_single_int_action(action):  # If action is a single integer
            action = np.int(action)

        return action

    def reset_eps(self):
        self.eps = self.epsstart

    def select_action(self, model: nn.Module, state, eps: float,
                      legal_actions: List[int], training: bool):
        sample = np.random.random()
        if not training or sample > eps:
            action = model(state)['a'].detach().cpu()
            return action.numpy()
        else:
            if legal_actions: return np.random.choice(legal_actions)
            else: return np.random.choice(range(self.algorithm.model.action_dim))

    def clone(self, training=None):
        clone = copy.deepcopy(self)
        clone.training = training
        return clone


def build_DQN_Agent(task, config, agent_name):
    kwargs = dict()
    """
    :param kwargs:
        "use_cuda": boolean to specify whether to use CUDA.
        "replay_capacity": int, capacity of the replay buffer to use.
        "min_capacity": int, minimal capacity before starting to learn.
        "batch_size": int, batch size to use [default: batch_size=256].
        "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
        "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
        "lr": float, learning rate [default: lr=1e-3].
        "tau": float, target network update rate.
        "gamma": float, Q-learning gamma rate.
        "preprocess": preprocessing function/transformation to apply to observations [default: preprocess=T.ToTensor()]
        "nbrTrainIteration": int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]
        "epsstart": starting value of the epsilong for the epsilon-greedy policy.
        "epsend": asymptotic value of the epsilon for the epsilon-greedy policy.
        "epsdecay": rate at which the epsilon of the epsilon-greedy policy decays.

        "dueling": boolean specifying whether to use Dueling Deep Q-Network architecture
        "double": boolean specifying whether to use Double Deep Q-Network algorithm.
        "nbr_actions": number of dimensions in the action space.
        "actfn": activation function to use in between each layer of the neural networks.
        "state_dim": number of dimensions in the state space.
    """

    preprocess = turn_into_single_element_batch

    kwargs['nbrTrainIteration'] = config['nbrTrainIteration']
    kwargs["nbr_actions"] = task.action_dim
    kwargs["actfn"] = LeakyReLU
    kwargs["state_dim"] = task.observation_dim

    # Create model architecture:
    body = FCBody(task.observation_dim)
    if config['dueling']:
        model = CategoricalDuelingDQNet(body=body,
                                        action_dim=task.action_dim)
    else:
        model = CategoricalDQNet(body=body,
                                 action_dim=task.action_dim,
                                 use_cuda=config['use_cuda'])
    model.share_memory()

    kwargs["dueling"] = config['dueling']
    kwargs["double"] = config['double']

    kwargs["use_cuda"] = config['use_cuda']

    kwargs["replay_capacity"] = float(config['memoryCapacity'])
    kwargs["min_capacity"] = float(config['min_memory'])
    kwargs["batch_size"] = int(config['batch_size'])
    kwargs["use_PER"] = config['use_PER']
    kwargs["PER_alpha"] = float(config['PER_alpha'])

    kwargs["lr"] = float(config['learning_rate'])
    kwargs["tau"] = float(config['tau'])
    kwargs["gamma"] = float(config['gamma'])

    kwargs["preprocess"] = preprocess

    kwargs['epsstart'] = float(config['epsstart'])
    kwargs['epsend'] = float(config['epsend'])
    kwargs['epsdecay'] = float(config['epsdecay'])

    DeepQNetwork_algo = DeepQNetworkAlgorithm(kwargs=kwargs, model=model)

    return DeepQNetworkAgent(name=agent_name, algorithm=DeepQNetwork_algo)
