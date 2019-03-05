import numpy as np

import torch
import torchvision.transforms as T

from ..replay_buffers import EXP, EXPPER
from ..networks import  LeakyReLU, ActorNN, CriticNN
from ..DDPG import DeepDeterministicPolicyGradientAlgorithm


class DDPGAgent():
    def __init__(self, algorithm):
        """
        :param algorithm: algorithm class to use to optimize the network(s).
        """

        self.algorithm = algorithm
        self.training = False
        self.preprocess_function = self.algorithm.kwargs["preprocess"]

        self.kwargs = algorithm.kwargs

        self.nbr_steps = 0

        self.name = self.kwargs['name']

    def getModel(self):
        return [self.algorithm.model_actor, self.algorithm.model_critic]

    def handle_experience(self, s, a, r, succ_s, done=False):
        hs = self.preprocess_function(s)
        hsucc = self.preprocess_function(succ_s)
        r = torch.ones(1)*r
        a = torch.from_numpy(a)
        experience = EXP(hs, a, hsucc, r, done)
        self.algorithm.handle_experience(experience=experience)

        if self.training:
            self.algorithm.train(iteration=self.kwargs['nbrTrainIteration'])

    def take_action(self, state):
        return self.act(state=self.preprocess_function(state), exploitation=not(self.training), exploration_noise=None)
        
    def reset_eps(self):
        pass

    def act(self, state, exploitation=True,exploration_noise=None) :
        if self.use_cuda :
            state = state.cuda()
        action = self.algorithm.model_actor( state).detach()
        
        if exploitation :
            return action.cpu().data.numpy()
        else :
            # exploration :
            if exploration_noise is not None :
                self.algorithm.noise.setSigma(exploration_noise)
            new_action = action.cpu().data.numpy() + self.algorithm.noise.sample()*self.algorithm.model_actor.action_scaler
            return new_action



    def clone(self, training=None, path=None):
        from ..agent_hook import AgentHook
        cloned = AgentHook(self, training=training, path=path)
        return cloned

class PreprocessFunction(object) :
    def __init__(self, hash_function=None,use_cuda=False):
        self.hash_function = hash_function
        self.use_cuda = use_cuda
    def __call__(self,x) :
        if self.hash_function is not None :
            x = self.hash_function(x)
        if self.use_cuda :
            return torch.from_numpy( x ).unsqueeze(0).type(torch.cuda.FloatTensor)
        else :
            return torch.from_numpy( x ).unsqueeze(0).type(torch.FloatTensor)


def build_DDPG_Agent(state_space_size=32,
                        action_space_size=3,
                        learning_rate=1e-3,
                        num_worker=1,
                        nbrTrainIteration=1,
                        action_scaler=1.0,
                        memoryCapacity=25e3,
                        use_PER=False,
                        alphaPER=0.7,
                        use_HER=False,
                        k_HER=2,
                        strategy_HER='future',
                        singlegoal_HER=False,
                        MIN_MEMORY=5e1,
                        use_cuda=False):
    kwargs = dict()
    """
    :param kwargs:
        "path": str specifying where to save the model(s).
        "use_cuda": boolean to specify whether to use CUDA.
        "replay_capacity": int, capacity of the replay buffer to use.
        "min_capacity": int, minimal capacity before starting to learn.
        "batch_size": int, batch size to use [default: batch_size=256].
        "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
        "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
        "lr": float, learning rate [default: lr=1e-3].
        "tau": float, target update rate [default: tau=1e-3].
        "gamma": float, Q-learning gamma rate [default: gamma=0.999].
        "preprocess": preprocessing function/transformation to apply to observations [default: preprocess=T.ToTensor()]
        "nbrTrainIteration": int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]
        "action_dim": number of dimensions in the action space.
        "state_dim": number of dimensions in the state space.
        "actfn": activation function to use in between each layer of the neural networks.
        
    """

    preprocess = PreprocessFunction(use_cuda=use_cuda)

    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 1e-3
    
    #HER :
    k = k_HER
    strategy = strategy_HER
    singlegoal = singlegoal_HER
    HER = {'k':k, 'strategy':strategy,'use_her':use_HER,'singlegoal':singlegoal}
    kwargs['HER'] = HER

    kwargs['nbrTrainIteration'] = nbrTrainIteration
    kwargs["action_dim"] = action_space_size
    kwargs["state_dim"] = state_space_size
    kwargs["action_scaler"] = action_scaler
    
    kwargs["actfn"] = LeakyReLU
    
    # Create model architecture:
    actor = ActorNN(state_dim=state_space_size,action_dim=action_space_size,action_scaler=action_scaler,HER=HER['use_her'],use_cuda=use_cuda)
    actor.share_memory()
    critic = CriticNN(state_dim=state_space_size,action_dim=action_space_size,HER=kwargs['HER']['use_her'],use_cuda=use_cuda)
    critic.share_memory()
    
    name = "DDPG"
    model_path = './'+name
    path=model_path

    kwargs['name'] = name
    kwargs["path"] = path
    kwargs["use_cuda"] = use_cuda

    kwargs["replay_capacity"] = memoryCapacity
    kwargs["min_capacity"] = MIN_MEMORY
    kwargs["batch_size"] = BATCH_SIZE
    kwargs["use_PER"] = use_PER
    kwargs["PER_alpha"] = alphaPER

    kwargs["lr"] = learning_rate
    kwargs["tau"] = TAU
    kwargs["gamma"] = GAMMA

    kwargs["preprocess"] = preprocess
    
    kwargs['replayBuffer'] = None

    DeepDeterministicPolicyGradient_algo = DeepDeterministicPolicyGradientAlgorithm(kwargs=kwargs, models={"actor":actor, "critic":critic})

    return DDPGAgent(algorithm=DeepDeterministicPolicyGradient_algo)
