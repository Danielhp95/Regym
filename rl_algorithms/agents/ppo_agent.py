import torch
import numpy as np
import copy

from ..networks import CategoricalActorCriticNet, GaussianActorCriticNet
from ..networks import FCBody
from ..networks import PreprocessFunctionToTorch, PreprocessFunction
from ..PPO import PPOAlgorithm

import torch.nn.functional as F


class PPOAgent(object):

    def __init__(self, name, algorithm):
        self.training = True
        self.algorithm = algorithm
        self.state_preprocessing = self.algorithm.kwargs['state_preprocess']
        self.handled_experiences = 0
        self.name = name
        self.nbr_actor = self.algorithm.kwargs['nbr_actor']

    def handle_experience(self, s, a, r, succ_s, done=False):
        non_terminal = torch.ones(1)*(1 - int(done))
        state = self.state_preprocessing(s)
        if isinstance(r, np.ndarray): 
            r = torch.from_numpy(r).float().view((1))
        else :
            r = torch.ones(1)*r
        r = r.cpu()
        a = torch.from_numpy(a).cpu().view((1,-1))

        current_prediction = self.algorithm.model(state)
        current_prediction = {k: v.detach().cpu().view((1,-1)) for k, v in current_prediction.items()}
        current_prediction['a'] = a 

        state = state.cpu().view((1,-1))
        self.algorithm.storage.add(current_prediction)
        self.algorithm.storage.add({'r': r, 'non_terminal': non_terminal, 's': state})

        self.handled_experiences += 1
        if self.training and self.handled_experiences >= self.algorithm.storage_capacity:
            next_state = self.state_preprocessing(succ_s)
            next_prediction = self.algorithm.model(next_state)
            next_prediction = {k: v.detach().cpu() for k, v in next_prediction.items()}
            self.algorithm.storage.add(next_prediction)            
            
            self.algorithm.train()
            self.handled_experiences = 0

    def take_action(self, s):
        state = self.state_preprocessing(s)
        current_prediction = self.algorithm.model(state)
        return current_prediction['a'].cpu().detach().numpy()

    def clone(self, training=None):
        clone = copy.deepcopy(self)
        clone.training = training
        return clone


def build_PPO_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for ppo agent
    :param agent_name: name of the agent
    :returns: PPOAgent adapted to be trained on :param: task under :param: config
    '''
    kwargs = config.copy()
    kwargs['state_preprocess'] = PreprocessFunctionToTorch(task.observation_dim, kwargs['use_cuda'])

    if task.action_type is 'Discrete' and task.observation_type is 'Discrete':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=FCBody(task.observation_dim, hidden_units=(64, 64), gate=F.leaky_relu),
                                          actor_body=None,
                                          critic_body=None)
        kwargs['state_preprocess'] = PreprocessFunction(task.observation_dim, kwargs['use_cuda'])

    if task.action_type is 'Continuous' and task.observation_type is 'Continuous':
        model = GaussianActorCriticNet(task.observation_dim, task.action_dim,
                                       phi_body=FCBody(task.observation_dim, hidden_units=(64, 64), gate=F.leaky_relu),
                                       actor_body=None,
                                       critic_body=None)

    model.share_memory()
    ppo_algorithm = PPOAlgorithm(kwargs, model)

    return PPOAgent(name=agent_name, algorithm=ppo_algorithm)
