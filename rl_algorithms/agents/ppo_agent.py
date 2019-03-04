import torch
import copy

from ..networks import CategoricalActorCriticNet, GaussianActorCriticNet
from ..networks import FCBody
from ..networks import PreprocessFunction
from ..PPO import PPOAlgorithm

import torch.nn.functional as F


class PPOAgent(object):

    def __init__(self, algorithm):
        self.training = True
        self.algorithm = algorithm
        self.state_preprocessing = self.algorithm.kwargs['state_preprocess']
        self.handled_experiences = 0
        self.name = 'PPO'

    def handle_experience(self, s, a, r, succ_s, done=False):
        non_terminal = torch.ones(1)*(1 - int(done))
        state = self.state_preprocessing(s)
        r = torch.ones(1)*r
        a = torch.from_numpy(a)

        self.algorithm.storage.add(self.current_prediction)
        self.algorithm.storage.add({'r': r, 'non_terminal': non_terminal, 's': state})

        self.handled_experiences += 1
        if self.training and self.handled_experiences >= self.algorithm.kwargs['horizon']:
            self.algorithm.train()
            self.handled_experiences = 0

    def take_action(self, state):
        state = self.state_preprocessing(state)
        self.current_prediction = self.algorithm.model(state)
        self.current_prediction = {k: v.detach() for k, v in self.current_prediction.items()}
        return self.current_prediction['a'].cpu().detach().numpy()

    def clone(self, training=None):
        clone = copy.deepcopy(self)
        clone.training = training
        return clone


def build_PPO_Agent(task, config):
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for ppo agent
    :returns: PPOAgent adapted to be trained on :param: task under :param: config
    '''
    kwargs = config.copy()
    kwargs['state_preprocess'] = PreprocessFunction(task.observation_dim, kwargs['use_cuda'])

    if task.action_type is 'Discrete' and task.observation_type is 'Discrete':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=FCBody(task.observation_dim, hidden_units=(64, 64), gate=F.leaky_relu),
                                          actor_body=None,
                                          critic_body=None)
    if task.action_type is 'Continuous' and task.observation_type is 'Continuous':
        model = GaussianActorCriticNet(task.observation_dim, task.action_dim,
                                       phi_body=FCBody(task.observation_dim, hidden_units=(64, 64), gate=F.leaky_relu),
                                       actor_body=None,
                                       critic_body=None)

    model.share_memory()
    ppo_algorithm = PPOAlgorithm(kwargs, model)

    return PPOAgent(algorithm=ppo_algorithm)
