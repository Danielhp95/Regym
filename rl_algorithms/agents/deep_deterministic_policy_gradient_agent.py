import torch
import numpy as np
import copy

from ..replay_buffers import EXP, EXPPER
from ..networks import  LeakyReLU, ActorNN, CriticNN, PreprocessFunctionToTorch
from ..DDPG import DeepDeterministicPolicyGradientAlgorithm


class DDPGAgent():
    def __init__(self, name, algorithm):
        """
        :param algorithm: algorithm class to use to optimize the network(s).
        """

        self.algorithm = algorithm
        self.training = True
        self.state_preprocessing = self.algorithm.kwargs["state_preprocessing"]
        self.kwargs = algorithm.kwargs
        self.nbr_steps = 0
        self.name = name

    def handle_experience(self, s, a, r, succ_s, done=False):
        hs = self.state_preprocessing(s).view((1,-1))
        hsucc = self.state_preprocessing(succ_s).view((1,-1))
        if isinstance(r, np.ndarray): 
            r = torch.from_numpy(r).float().view((1))
        else :
            r = torch.ones(1)*r
        a = torch.from_numpy(a).cpu().view((1,-1))

        experience = EXP(hs, a, hsucc, r, done)
        self.algorithm.handle_experience(experience=experience)

        if self.training:
            self.algorithm.train(iteration=self.kwargs['nbrTrainIteration'])

    def take_action(self, state):
        return self.act(state=self.state_preprocessing(state), exploitation=not(self.training), exploration_noise=None)
        
    def reset_eps(self):
        pass

    def act(self, state, exploitation=True,exploration_noise=None) :
        action = self.algorithm.model_actor( state).detach()
        
        if exploitation :
            return action.cpu().data.numpy()
        else :
            # exploration :
            if exploration_noise is not None :
                self.algorithm.noise.setSigma(exploration_noise)
            new_action = action.cpu().data.numpy() + self.algorithm.noise.sample()*self.algorithm.model_actor.action_scaler
            return new_action



    def clone(self, training=None):
        clone = copy.deepcopy(self)
        if training is not None:
            clone.training = training
        return clone

def build_DDPG_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for ppo agent
    :param agent_name: name of the agent
    :returns: DDPGAgent adapted to be trained on :param: task under :param: config
    '''
    
    kwargs = config.copy()
    """
    :param kwargs:
        "use_cuda": boolean to specify whether to use CUDA.
        "replay_capacity": int, capacity of the replay buffer to use.
        "min_capacity": int, minimal capacity before starting to learn.
        "batch_size": int, batch size to use [default: batch_size=256].
        "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
        "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
        "use_HER": False
        "HER_k": 2
        "HER_strategy": 'future'
        "HER_use_singlegoal": False 
        "lr": float, learning rate [default: lr=1e-3].
        "tau": float, target update rate [default: tau=1e-3].
        "gamma": float, Q-learning gamma rate [default: gamma=0.999].
        "nbrTrainIteration": int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]
    Elements added:
        "state_preprocessing": preprocessing function/transformation to apply to observations [default: preprocess=T.ToTensor()]
        "action_scaler": float, factor with which the output of the actor network is multiplied.
        "action_dim": number of dimensions in the action space.
        "state_dim": number of dimensions in the state space.
        "actfn": activation function to use in between each layer of the neural networks.
        
    """
    kwargs['replay_capacity'] = float(kwargs['replay_capacity'])
    kwargs['min_capacity'] = float(kwargs['min_capacity'])
    kwargs['state_preprocessing'] = PreprocessFunctionToTorch(task.observation_dim, kwargs['use_cuda'])

    #HER :
    HER = {'k':kwargs['HER_k'], 'strategy':kwargs['HER_strategy'],'use_her':kwargs['use_HER'],'singlegoal':kwargs['HER_use_singlegoal']}
    kwargs['HER'] = HER

    kwargs["action_dim"] = task.action_dim
    kwargs["state_dim"] = task.observation_dim
    kwargs["actfn"] = LeakyReLU
    
    # Create model architecture:
    actor = ActorNN(state_dim=task.observation_dim,action_dim=task.action_dim,action_scaler=kwargs['action_scaler'],HER=HER['use_her'],use_cuda=kwargs['use_cuda'])
    actor.share_memory()
    critic = CriticNN(state_dim=task.observation_dim,action_dim=task.action_dim,HER=kwargs['HER']['use_her'],use_cuda=kwargs['use_cuda'])
    critic.share_memory()
    
    kwargs['replayBuffer'] = None

    DeepDeterministicPolicyGradient_algo = DeepDeterministicPolicyGradientAlgorithm(kwargs=kwargs, models={"actor":actor, "critic":critic})

    return DDPGAgent(name=agent_name, algorithm=DeepDeterministicPolicyGradient_algo)
