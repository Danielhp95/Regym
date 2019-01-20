import os
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
			assert(path is not None)
			self.path=path
			self.type = AgentType.DQN 
			self.kwargs = dict()
			for name in agent.kwargs:
				if 'model' in name :
					continue
				else :
					self.kwargs[name] = agent.kwargs[name]
			# Saving CPU state_dict:
			print('SAVING : ',self.path)
			torch.save( agent.getModel().cpu().state_dict(), self.path)
			print('SAVING : {} :: OK'.format(self.path) )
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
			if use_cuda is not None :
				self.kwargs['use_cuda'] = use_cuda
				self.kwargs['preprocess'].use_cuda = use_cuda
			# Init Model:
			if self.kwargs['dueling']:
				model = DuelingDQN(nbr_actions=self.kwargs['nbr_actions'],actfn=self.kwargs['actfn'],useCNN=self.kwargs['useCNN'],use_cuda=False)
			else :
				model = DQN(nbr_actions=self.kwargs['nbr_actions'],actfn=self.kwargs['actfn'],useCNN=self.kwargs['useCNN'],use_cuda=False)
			# Loading CPU state_dict:
			print('LOADING : ',self.path)
			model.load_state_dict( torch.load(self.path) )
			print('LOADING : {} :: OK'.format(self.path) )
			if self.kwargs['use_cuda'] :
				model = model.cuda()
			# Init Algorithm
			kwargs = copy.deepcopy(self.kwargs)
			kwargs['model'] = model 
			if self.kwargs['double']:
				algorithm = DoubleDeepQNetworkAlgorithm(kwargs=kwargs)
			else :
				algorithm = DeepQNetworkAlgorithm(kwargs=kwargs)
			#TODO : decide whether to clone the replayBuffer or not:
			#cloned.replayBuffer = self.replayBuffer
			# Init Agent
			agent = DeepQNetworkAgent(network=None,algorithm=algorithm)
			if training is not None :
				agent.training = training
			return agent
		else :
			# MixedStrategyAgent
			return self.agent 

	def clone(self,training=None, path=None):
		cloned = copy.deepcopy(self)
		cloned.training = training
		return cloned





# TODO : remove DQNA2Queeue

class DeepQNetworkAgent2Queue():
    def __init__(self, dqnAgent, training=False, use_cuda=None):
        self.name = dqnAgent.name 
        self.training = training
        
        if isinstance(dqnAgent,DeepQNetworkAgent):                
            self.kwargs = dict()
            for name in dqnAgent.kwargs:
                if 'model' in name :
                    continue
                else :
                    self.kwargs[name] = dqnAgent.kwargs[name]
            
            if use_cuda is not None : 
                self.kwargs['use_cuda'] = use_cuda
                self.kwargs['preprocess'].use_cuda = use_cuda

            self.kwargs['model'] = dqnAgent.kwargs["model"].state_dict()
            for name in self.kwargs["model"] :
                self.kwargs["model"][name] = self.kwargs["model"][name].cpu().numpy()
        else :
            # cloning :
            self.kwargs = copy.deepcopy(dqnAgent.kwargs) 

    def queue2policy(self, use_cuda=None):
        for name in self.kwargs["model"] :
            self.kwargs["model"][name] = torch.from_numpy( self.kwargs["model"][name] )
        
        if use_cuda is not None : 
            self.kwargs['use_cuda'] = use_cuda
            self.kwargs['preprocess'].use_cuda = use_cuda
        
        if self.kwargs['dueling']:
            model = DuelingDQN(nbr_actions=self.kwargs['nbr_actions'],actfn=self.kwargs['actfn'],useCNN=self.kwargs['useCNN'],use_cuda=False)
        else :
            model = DQN(nbr_actions=self.kwargs['nbr_actions'],actfn=self.kwargs['actfn'],useCNN=self.kwargs['useCNN'],use_cuda=False)
        
        model.load_state_dict(self.kwargs["model"])
        if self.kwargs['use_cuda'] :
            model = model.cuda()

        self.kwargs['model'] = model 

        if self.kwargs['double']:
            algorithm = DOubleDeepQNetworkAlgorithm(kwargs=self.kwargs)
        else :
            algorithm = DeepQNetworkAlgorithm(kwargs=self.kwargs)
        
        #TODO : decide whether to clone the replayBuffer or not:
        #cloned.replayBuffer = self.replayBuffer
        
        policy = DeepQNetworkAgent(network=None,algorithm=algorithm)
        policy.training = self.training
        
        return policy

    def clone(self):
        return DeepQNetworkAgent2Queue(self)