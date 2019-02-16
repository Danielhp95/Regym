import numpy as np
import random 
import torch
import torchvision.transforms as T

from ..replay_buffers import EXP, EXPPER
from ..networks import  LeakyReLU, DQN, DuelingDQN
from ..DQN import DeepQNetworkAlgorithm, DoubleDeepQNetworkAlgorithm 


class DeepQNetworkAgent():
    def __init__(self, algorithm):
        """
        :param algorithm: algorithm class to use to optimize the network.
        """

        self.algorithm = algorithm
        self.training = False
        self.preprocessing_function = self.algorithm.kwargs["preprocess"]

        self.kwargs = algorithm.kwargs

        self.epsend = self.kwargs['epsend']
        self.epsstart = self.kwargs['epsstart']
        self.epsdecay = self.kwargs['epsdecay']
        self.nbr_steps = 0

        self.name = self.kwargs['name']

    def getModel(self):
        return self.algorithm.model

    def handle_experience(self, s, a, r, succ_s, done=False):
        hs = self.preprocessing_function(s)
        hsucc = self.preprocessing_function(succ_s)
        r = torch.ones(1)*r
        a = torch.from_numpy(a)
        experience = EXP(hs, a, hsucc, r, done)
        self.algorithm.handle_experience(experience=experience)

        if self.training:
            self.algorithm.train(iteration=self.kwargs['nbrTrainIteration'])

    def take_action(self, state):
        self.nbr_steps += 1
        self.eps = self.epsend + (self.epsstart-self.epsend) * np.exp(-1.0 * self.nbr_steps / self.epsdecay)
        action, qsa = self.select_action(model=self.algorithm.model, state=self.preprocessing_function(state), eps=self.eps)
        return action

    def reset_eps(self):
        self.eps = self.epsstart

    def select_action(self,model,state,eps) :
        sample = np.random.random()
        if sample > eps :
            output = model( state ).cpu().data
            qsa, action = output.max(1)
            action = action.view(1,1)
            qsa = output.max(1)[0].view(1,1)[0,0]
            return action.numpy(), qsa
        else :
            random_action = torch.LongTensor( [[random.randrange(self.algorithm.model.nbr_actions) ] ] )
            return random_action.numpy(), 0.0


    def clone(self, training=None, path=None):
        from ..agent_hook import AgentHook
        cloned = AgentHook(self, training=training, path=path)
        return cloned

class PreprocessFunction(object) :
    def __init__(self, state_space_size,use_cuda=False):
        self.state_space_size = state_space_size
        self.use_cuda = use_cuda
    def __call__(self,x) :
        x = np.concatenate(x, axis=None)
        if self.use_cuda :
            return torch.from_numpy( x ).unsqueeze(0).type(torch.cuda.FloatTensor)
        else :
            return torch.from_numpy( x ).unsqueeze(0).type(torch.FloatTensor)


def build_DQN_Agent(state_space_size=32,
                        action_space_size=3,
                        learning_rate=6.25e-5,
                        double=False,
                        dueling=False,
                        num_worker=1,
                        nbrTrainIteration=32,
                        memoryCapacity = 25e3,
                        use_PER=False,
                        alphaPER=0.7,
                        MIN_MEMORY=5e1,
                        epsstart=0.8,
                        epsend=0.05,
                        epsdecay=1e3,
                        use_cuda=False):
    kwargs = dict()
    """
    :param kwargs:
        "model": model of the agent to use/optimize in this algorithm.
        "path": str specifying where to save the model(s).
        "use_cuda": boolean to specify whether to use CUDA.
        "replay_capacity": int, capacity of the replay buffer to use.
        "min_capacity": int, minimal capacity before starting to learn.
        "batch_size": int, batch size to use [default: batch_size=256].
        "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
        "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
        "lr": float, learning rate [default: lr=1e-3].
        "tau": float, target update rate [default: tau=1e-2].
        "gamma": float, Q-learning gamma rate [default: gamma=0.999].
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

    preprocess = PreprocessFunction( state_space_size=state_space_size,use_cuda=use_cuda)
    
    kwargs['nbrTrainIteration'] = nbrTrainIteration
    kwargs["nbr_actions"] = action_space_size
    kwargs["actfn"] = LeakyReLU
    kwargs["state_dim"] = state_space_size
    # Create model architecture:
    if dueling :
        model = DuelingDQN(state_space_size, action_space_size, use_cuda=use_cuda)
    else :
        model = DQN(state_space_size, action_space_size, use_cuda=use_cuda)
    model.share_memory()

    kwargs["model"] = model
    kwargs["dueling"] = dueling
    kwargs["double"] = double

    BATCH_SIZE = 256
    GAMMA = 0.99
    TAU = 1e-2
    lr = 1e-3
    
    name = "DQN"
    if dueling : name = 'Dueling'+name
    if double : name = 'Double'+name
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

    kwargs["lr"] = lr
    kwargs["tau"] = TAU
    kwargs["gamma"] = GAMMA

    kwargs["preprocess"] = preprocess
    
    kwargs['epsstart'] = epsstart
    kwargs['epsend'] = epsend
    kwargs['epsdecay'] = epsdecay

    kwargs['replayBuffer'] = None

    DeepQNetwork_algo = DoubleDeepQNetworkAlgorithm(kwargs=kwargs) if dueling else DeepQNetworkAlgorithm(kwargs=kwargs)

    return DeepQNetworkAgent(algorithm=DeepQNetwork_algo)

