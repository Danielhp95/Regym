import torch
from .networks import DQN, DuelingDQN, ActorNN, CriticNN
from .DQN import DeepQNetworkAlgorithm, DoubleDeepQNetworkAlgorithm
from .DDPG import DeepDeterministicPolicyGradientAlgorithm
from .TQL import TabularQLearningAlgorithm
from .agents import TabularQLearningAgent, DeepQNetworkAgent, DDPGAgent
from enum import Enum
import copy

AgentType = Enum("AgentType", "DQN TQL DDPG PPO MixedStrategyAgent")


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

            if isinstance(agent, DeepQNetworkAgent):
                    self.path = path
                    self.type = AgentType.DQN
                    self.kwargs = dict()
                    for name in agent.kwargs:
                            if 'model' in name:
                                    continue
                            else:
                                    self.kwargs[name] = agent.kwargs[name]
                    # Saving CPU state_dict and RB:
                    if self.path is not None:
                            torch.save(agent.getModel().cpu().state_dict(), self.path)
                            agent.algorithm.replayBuffer.save(self.path)
                    else:
                            self.agent = copy.deepcopy(agent)
            elif isinstance(agent, TabularQLearningAgent):
                    self.type = AgentType.TQL
                    self.agent = agent
            
            if isinstance(agent, DDPGAgent):
                    self.path = path
                    self.type = AgentType.DDPG
                    self.kwargs = dict()
                    for name in agent.kwargs:
                            if 'model' in name:
                                    continue
                            else:
                                    self.kwargs[name] = agent.kwargs[name]
                    # Saving CPU state_dict and RB:
                    if self.path is not None:
                            model_actor, model_critic = agent.getModel()
                            torch.save(model_actor.cpu().state_dict(), self.path+'actor')
                            torch.save(model_critic.cpu().state_dict(), self.path+'critic')
                            agent.algorithm.replayBuffer.save(self.path)
                    else:
                            self.agent = copy.deepcopy(agent)        
            else:
                    self.type = AgentType.MixedStrategyAgent
                    self.agent = agent
            # TODO : other agents implementation

    def __call__(self, training=None, use_cuda=None):
            if self.type == AgentType.TQL:
                    return copy.deepcopy(self.agent)
            if self.type == AgentType.DQN:
                    if self.path is None:
                            return self.agent
                    else:
                            if use_cuda is not None:
                                    self.kwargs['use_cuda'] = use_cuda
                                    self.kwargs['preprocess'].use_cuda = use_cuda
                            # Init Model:
                            if self.kwargs['dueling']:
                                    model = DuelingDQN(state_dim=self.kwargs['state_dim'],nbr_actions=self.kwargs['nbr_actions'], actfn=self.kwargs['actfn'], use_cuda=False)
                            else:
                                    model = DQN(state_dim=self.kwargs['state_dim'],nbr_actions=self.kwargs['nbr_actions'], actfn=self.kwargs['actfn'], use_cuda=False)
                            # Loading CPU state_dict:
                            model.load_state_dict(torch.load(self.path))
                            if self.kwargs['use_cuda']:
                                    model = model.cuda()
                            # Init Algorithm
                            kwargs = copy.deepcopy(self.kwargs)
                            kwargs['model'] = model
                            if self.kwargs['double']:
                                    algorithm = DoubleDeepQNetworkAlgorithm(kwargs=kwargs)
                            else:
                                    algorithm = DeepQNetworkAlgorithm(kwargs=kwargs)
                            # Init Agent
                            agent = DeepQNetworkAgent(algorithm=algorithm)
                            if training is not None:
                                    agent.training = training
                            if agent.training:
                                    agent.algorithm.replayBuffer.load(self.path)
                            return agent
            elif self.type == AgentType.DDPG:
                    if self.path is None:
                            return self.agent
                    else:
                            if use_cuda is not None:
                                    self.kwargs['use_cuda'] = use_cuda
                            # Init Model:
                            model_actor = ActorNN(state_dim=self.kwargs['state_dim'],action_dim=self.kwargs['action_dim'],action_scaler=self.kwargs['action_scaler'], actfn=self.kwargs['actfn'], use_cuda=False)
                            model_critic = CriticNN(state_dim=self.kwargs['state_dim'],action_dim=self.kwargs['action_dim'],HER=self.kwargs['HER']['use_her'], actfn=self.kwargs['actfn'], use_cuda=False)
                            # Loading CPU state_dict:
                            model_actor.load_state_dict(torch.load(self.path+'actor'))
                            model_critic.load_state_dict(torch.load(self.path+'critic'))
                            if self.kwargs['use_cuda']:
                                    model_actor = model_actor.cuda()
                                    model_critic = model_critic.cuda()
                            # Init Algorithm
                            kwargs = copy.deepcopy(self.kwargs)
                            kwargs['model_actor'] = model_actor
                            kwargs['model_critic'] = model_critic
                            algorithm = DeepDeterministicPolicyGradientAlgorithm(kwargs=kwargs)
                            # Init Agent
                            agent = DDPGAgent(algorithm=algorithm)
                            if training is not None:
                                    agent.training = training
                            if agent.training:
                                    agent.algorithm.replayBuffer.load(self.path)
                            return agent
            else:
                    # MixedStrategyAgent
                    return self.agent

    def clone(self, training=None, path=None):
            cloned = copy.deepcopy(self)
            cloned.training = training
            return cloned
