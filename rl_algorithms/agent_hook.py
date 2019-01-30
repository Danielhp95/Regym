import torch
from .deep_q_network import DeepQNetworkAgent, DQN, DuelingDQN, DeepQNetworkAlgorithm, DoubleDeepQNetworkAlgorithm
from .tabular_q_learning import TabularQLearningAgent
from .gym_rock_paper_scissors_agent import MixedStrategyAgent
from enum import Enum
import copy

AgentType = Enum("AgentType", "DQN TQL PPO MixedStrategyAgent")


class AgentHook():
    def __init__(self, agent, training=None, path=None):
            """
            AgentHook is to be used as a saver and loader object for agents prior to or following their transportation from
            a training scheme to a benchmarking scheme etc...

            :param agent: any agent to hang for transportation between processes.
            :param path: path where to save the current agent.
            """

            self.name = agent.name
            self.training = training
            self.type = None

            if isinstance(agent,DeepQNetworkAgent):
                    self.path=path
                    self.type = AgentType.DQN
                    self.kwargs = dict()
                    for name in agent.kwargs:
                            if 'model' in name :
                                    continue
                            else :
                                    self.kwargs[name] = agent.kwargs[name]
                    # Saving CPU state_dict and RB:
                    if self.path is not None :
                            torch.save( agent.getModel().cpu().state_dict(), self.path)
                            agent.algorithm.replayBuffer.save(self.path)
                    else :
                            self.agent = copy.deepcopy(agent)
            elif isinstance(agent,TabularQLearningAgent):
                    self.type = AgentType.TQL
                    self.agent = agent
            else :
                    self.type = AgentType.MixedStrategyAgent
                    self.agent = agent
            #TODO : other agents implementation

    def __call__(self, training=None, use_cuda=None):
            if self.type == AgentType.TQL :
                    return copy.deepcopy(self.agent)
            if self.type == AgentType.DQN :
                    if self.path is None :
                            return self.agent
                    else :
                            if use_cuda is not None :
                                    self.kwargs['use_cuda'] = use_cuda
                                    self.kwargs['preprocess'].use_cuda = use_cuda
                            # Init Model:
                            if self.kwargs['dueling']:
                                    model = DuelingDQN(nbr_actions=self.kwargs['nbr_actions'],actfn=self.kwargs['actfn'],useCNN=self.kwargs['useCNN'],use_cuda=False)
                            else :
                                    model = DQN(nbr_actions=self.kwargs['nbr_actions'],actfn=self.kwargs['actfn'],useCNN=self.kwargs['useCNN'],use_cuda=False)
                            # Loading CPU state_dict:
                            model.load_state_dict( torch.load(self.path) )
                            if self.kwargs['use_cuda'] :
                                    model = model.cuda()
                            # Init Algorithm
                            kwargs = copy.deepcopy(self.kwargs)
                            kwargs['model'] = model
                            if self.kwargs['double']:
                                    algorithm = DoubleDeepQNetworkAlgorithm(kwargs=kwargs)
                            else :
                                    algorithm = DeepQNetworkAlgorithm(kwargs=kwargs)
                            # Init Agent
                            agent = DeepQNetworkAgent(network=None,algorithm=algorithm)
                            if training is not None :
                                    agent.training = training
                            if agent.training :
                                    agent.algorithm.replayBuffer.load(self.path)
                            return agent
            else :
                    # MixedStrategyAgent
                    return self.agent

    def clone(self,training=None, path=None):
            cloned = copy.deepcopy(self)
            cloned.training = training
            return cloned
