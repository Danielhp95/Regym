import torch
from .networks import DQN, DuelingDQN, ActorNN, CriticNN
from .DQN import DeepQNetworkAlgorithm, DoubleDeepQNetworkAlgorithm
from .DDPG import DeepDeterministicPolicyGradientAlgorithm
from .agents import TabularQLearningAgent, DeepQNetworkAgent, DDPGAgent, MixedStrategyAgent
from enum import Enum
import copy

AgentType = Enum("AgentType", "DQN TQL DDPG PPO MixedStrategyAgent")


class AgentHook_deprec():
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

        if isinstance(agent, DeepQNetworkAgent): self.init_dqn_agent(agent, training, path)
        elif isinstance(agent, TabularQLearningAgent): self.init_tql_agent(agent, training, path)
        elif isinstance(agent, DDPGAgent): self.init_ddpg_agent(agent, training, path)
        elif isinstance(agent, PPOAgent): self.init_ppo_agent(agent, training, path)
        elif isinstance(agent, MixedStrategyAgent): self.init_mixedstrategy_agent(agent, training, path)
        else: raise ValueError('Unknown AgentType {}, valid types are: {}'.format(type(agent), [t for t in AgentType]))

    def __call__(self, training=None, use_cuda=None):
        if self.type == AgentType.TQL: return copy.deepcopy(self.agent)
        if self.type == AgentType.DQN: return self.call_dqn(training, use_cuda)
        if self.type == AgentType.DDPG: return self.call_ddpg(training, use_cuda)
        if self.type == AgentType.PPO: return self.call_ppo(training, use_cuda)
        if self.type == AgentType.MixedStrategyAgent: return self.agent
        else: raise ValueError('Unknown AgentType {}, valid types are: {}'.format(self.type, [t for t in AgentType]))

    def init_dqn_agent(self, agent, training, path):
        self.path = path
        self.type = AgentType.DQN
        self.kwargs = dict()
        for name in agent.kwargs:
            if 'model' not in name: self.kwargs[name] = agent.kwargs[name]
        # Saving CPU state_dict and RB:
        if self.path is not None:
            torch.save(agent.getModel().cpu().state_dict(), self.path)
            agent.algorithm.replayBuffer.save(self.path)
        else:
            self.agent = copy.deepcopy(agent)

    def init_tql_agent(self, agent, training, path):
        self.type = AgentType.TQL
        self.agent = agent

    def init_ddpg_agent(self, agent, training, path):
        self.path = path
        self.type = AgentType.DDPG
        self.kwargs = dict()
        for name in agent.kwargs:
            if 'model' not in name: self.kwargs[name] = agent.kwargs[name]
        # Saving CPU state_dict and Replay Buffer:
        if self.path is not None:
            model_actor, model_critic = agent.getModel()
            torch.save(model_actor.cpu().state_dict(), self.path+'actor')
            torch.save(model_critic.cpu().state_dict(), self.path+'critic')
            agent.algorithm.replayBuffer.save(self.path)
        else:
            self.agent = copy.deepcopy(agent)

    def init_ppo_agent(self, agent, training, path):
        self.path = path
        self.type = AgentType.PPO
        self.kwargs = dict()
        for name in agent.kwargs:
            if 'model' not in name: self.kwargs[name] = agent.kwargs[name]
        # Saving CPU state_dict and Replay Buffer:
        if self.path is not None:
            model = agent.getModel()
            torch.save(model.cpu().state_dict(), self.path)
        else:
            self.agent = copy.deepcopy(agent)

    def init_mixedstrategy_agent(self, agent, training, path):
        self.type = AgentType.MixedStrategyAgent
        self.agent = agent

    def call_ppo(self, training, use_cuda):
        # TODO Test ppo as a whole
        if self.path is None: return self.agent

        if use_cuda is not None:
            self.kwargs['use_cuda'] = use_cuda
        # Init Model:
        kwargs['state_preprocess'] = PreprocessFunction(task.observation_dim, kwargs['use_cuda'])

        raise NotImplementedError('')

        #TODO : build the model with the correct argument from the ?task information?
        if self.kwargs['task_action_type'] is 'Discrete' and self.kwargs['task_observation_type'] is 'Discrete':
            model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                                        phi_body=FCBody(task.observation_dim, hidden_units=(64, 64), gate=F.leaky_relu),
                                                        actor_body=None,
                                                        critic_body=None)
        if task.action_type is 'Continuous' and task.observation_type is 'Continuous':
            kwargs['model'] = GaussianActorCriticNet(task.observation_dim, task.action_dim,
                                                     phi_body=FCBody(task.observation_dim, hidden_units=(64, 64), gate=F.leaky_relu),
                                                     actor_body=None,
                                                     critic_body=None)

        #model_actor = ActorNN(state_dim=self.kwargs['state_dim'], action_dim=self.kwargs['action_dim'], action_scaler=self.kwargs['action_scaler'], actfn=self.kwargs['actfn'], use_cuda=False)
        #model_critic = CriticNN(state_dim=self.kwargs['state_dim'], action_dim=self.kwargs['action_dim'], HER=self.kwargs['HER']['use_her'], actfn=self.kwargs['actfn'], use_cuda=False)
        
        # Loading CPU state_dict:
        model.load_state_dict(torch.load(self.path))
        if self.kwargs['use_cuda']:
            model = model.cuda()
        
        # Init Algorithm
        kwargs = copy.deepcopy(self.kwargs)
        kwargs['model'] = model
        algorithm = PPOAlgorithm(kwargs)
        
        # Init Agent
        agent = PPOAgent(algorithm=algorithm)
        if training is not None: agent.training = training
        return agent

    def call_dqn(self, training, use_cuda):
        if self.path is None: return self.agent
        if use_cuda is not None:
            self.kwargs['use_cuda'] = use_cuda
            self.kwargs['preprocess'].use_cuda = use_cuda
        # Init Model:
        if self.kwargs['dueling']:
            model = DuelingDQN(state_dim=self.kwargs['state_dim'], nbr_actions=self.kwargs['nbr_actions'], actfn=self.kwargs['actfn'], use_cuda=False)
        else:
            model = DQN(state_dim=self.kwargs['state_dim'], nbr_actions=self.kwargs['nbr_actions'], actfn=self.kwargs['actfn'], use_cuda=False)
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

    def call_ddpg(self, training, use_cuda):
        # TODO Test ddpg as a whole
        if self.path is None: return self.agent

        if use_cuda is not None:
            self.kwargs['use_cuda'] = use_cuda
        # Init Model:
        model_actor = ActorNN(state_dim=self.kwargs['state_dim'], action_dim=self.kwargs['action_dim'], action_scaler=self.kwargs['action_scaler'], actfn=self.kwargs['actfn'], use_cuda=False)
        model_critic = CriticNN(state_dim=self.kwargs['state_dim'], action_dim=self.kwargs['action_dim'], HER=self.kwargs['HER']['use_her'], actfn=self.kwargs['actfn'], use_cuda=False)
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
        if training is not None: agent.training = training
        if agent.training: agent.algorithm.replayBuffer.load(self.path)
        return agent

    def clone(self, training=None, path=None):
        cloned = copy.deepcopy(self)
        cloned.training = training
        return cloned




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

        self.path = path
        torch.save(agent, self.path)

    def __call__(self, training=None, use_cuda=None):
        agent = torch.load(self.path)
        if training is not None :
            self.training = training
        agent.training = self.training
        if use_cuda is not None :
            agent.algorithm.kwargs['use_cuda'] = use_cuda
            if agent.algorithm.kwargs['use_cuda']:
                agent.algorithm.model = agent.algorithm.model.cuda()
            else:
                agent.algorithm.model = agent.algorithm.model.cpu()

        return agent 
    
    def clone(self, training=None, path=None):
        cloned = copy.deepcopy(self)
        cloned.training = training
        return cloned
