import torch
from .networks import DQN, DuelingDQN, ActorNN, CriticNN
from .agents import TabularQLearningAgent, DeepQNetworkAgent, DDPGAgent, PPOAgent, MixedStrategyAgent
from enum import Enum
import copy

AgentType = Enum("AgentType", "DQN TQL DDPG PPO MixedStrategyAgent")


class AgentHook():
    def __init__(self, agent, training=None, path=None, use_cuda=None):
        """
        AgentHook is to be used as a saver and loader object for agents prior to or following their transportation from
        a training scheme to a benchmarking scheme etc...

        :param agent: any agent to hang for transportation between processes.
        :param path: path where to save the current agent.
        """

        self.name = agent.name
        self.training = training
        self.path = path
        self.use_cuda = use_cuda
        self.type = None

        if isinstance(agent, DeepQNetworkAgent): self.init_dqn_agent(agent, training)
        elif isinstance(agent, TabularQLearningAgent): self.init_tql_agent(agent, training)
        elif isinstance(agent, DDPGAgent): self.init_ddpg_agent(agent, training)
        elif isinstance(agent, PPOAgent): self.init_ppo_agent(agent, training)
        elif isinstance(agent, MixedStrategyAgent): self.init_mixedstrategy_agent(agent, training)
        else: raise ValueError('Unknown AgentType {}, valid types are: {}'.format(type(agent), [t for t in AgentType]))

    def __call__(self, training=None, use_cuda=None):
        if self.type == AgentType.TQL: return copy.deepcopy(self.agent)
        if self.type == AgentType.DQN: return self.call_dqn(training, use_cuda)
        if self.type == AgentType.DDPG: return self.call_ddpg(training, use_cuda)
        if self.type == AgentType.PPO: return self.call_ppo(training, use_cuda)
        if self.type == AgentType.MixedStrategyAgent: return self.agent
        else: raise ValueError('Unknown AgentType {}, valid types are: {}'.format(self.type, [t for t in AgentType]))

    def init_dqn_agent(self, agent, training):
        self.type = AgentType.DQN
        
        agent.algorithm.model.cpu()
        agent.algorithm.target_model.cpu()

        if self.path is not None :
            torch.save(agent, self.path)
        else :
            self.agent = agent
    
    def init_tql_agent(self, agent, training):
        self.type = AgentType.TQL
        self.agent = agent

        if self.path is not None :
            torch.save(agent, self.path)
        else :
            self.agent = agent

    def init_ddpg_agent(self, agent, training):
        self.type = AgentType.DDPG
        
        agent.algorithm.model_actor.cpu()
        agent.algorithm.model_critic.cpu()

        if self.path is not None :
            torch.save(agent, self.path)
        else :
            self.agent = agent

    def init_ppo_agent(self, agent, training):
        self.type = AgentType.PPO

        agent.algorithm.model.cpu()

        if self.path is not None :
            torch.save(agent, self.path)
        else :
            self.agent = agent
        
    def init_mixedstrategy_agent(self, agent, training):
        self.type = AgentType.MixedStrategyAgent
        self.agent = agent

    def call_ppo(self, training, use_cuda):
        if self.path is not None :
            agent = torch.load(self.path)
        else :
            agent = self.agent 

        if training is not None :
            self.training = training
        agent.training = self.training

        if self.use_cuda is None :
            self.use_cuda = agent.algorithm.kwargs['use_cuda']
        if use_cuda is not None :
            self.use_cuda = use_cuda
        agent.algorithm.kwargs['use_cuda'] = self.use_cuda
        
        if agent.algorithm.kwargs['use_cuda']:
            agent.algorithm.model = agent.algorithm.model.cuda()
        else:
            agent.algorithm.model = agent.algorithm.model.cpu()
        
        return agent

    def call_dqn(self, training, use_cuda):
        if self.path is not None :
            agent = torch.load(self.path)
        else :
            agent = self.agent 

        if training is not None :
            self.training = training
        agent.training = self.training

        if self.use_cuda is None :
            self.use_cuda = agent.algorithm.kwargs['use_cuda']
        if use_cuda is not None :
            self.use_cuda = use_cuda
        agent.algorithm.kwargs['use_cuda'] = self.use_cuda
        
        if agent.algorithm.kwargs['use_cuda']:
            agent.algorithm.model = agent.algorithm.model.cuda()
            agent.algorithm.target_model = agent.algorithm.target_model.cuda()
        else:
            agent.algorithm.model = agent.algorithm.model.cpu()
            agent.algorithm.target_model = agent.algorithm.target_model.cpu()
        
        return agent

    def call_ddpg(self, training, use_cuda):
        if self.path is not None :
            agent = torch.load(self.path)
        else :
            agent = self.agent 

        if training is not None :
            self.training = training
        agent.training = self.training

        if self.use_cuda is None :
            self.use_cuda = agent.algorithm.kwargs['use_cuda']
        if use_cuda is not None :
            self.use_cuda = use_cuda
        agent.algorithm.kwargs['use_cuda'] = self.use_cuda
        
        if agent.algorithm.kwargs['use_cuda']:
            agent.algorithm.model_actor = agent.algorithm.model_actor.cuda()
            agent.algorithm.model_critic = agent.algorithm.model_critic.cuda()
        else:
            agent.algorithm.model_actor = agent.algorithm.model_actor.cpu()
            agent.algorithm.model_critic = agent.algorithm.model_critic.cpu()
        
        return agent

    def clone(self, training=None, path=None):
        cloned = copy.deepcopy(self)
        cloned.training = training
        return cloned