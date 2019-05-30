import torch
import copy

from ..networks import CategoricalActorCriticNet, GaussianActorCriticNet
from ..networks import FCBody, LSTMBody
from ..networks import PreprocessFunction
from ..PPO import PPOAlgorithm

import torch.nn.functional as F


class PPOAgent(object):

    def __init__(self, name, algorithm):
        self.training = True
        self.algorithm = algorithm
        self.state_preprocessing = self.algorithm.kwargs['state_preprocess']
        self.handled_experiences = 0
        self.name = name

        self.recurrent = False 
        self.rnn_states = None 
        self.rnn_keys = [ key for key,value in self.algorithm.kwargs.items() if isinstance(value, str) and 'RNN' in value]
        if len(self.rnn_keys):
            self.recurrent = True
            self._reset_rnn_states()
        
    def _reset_rnn_states(self):
        self.rnn_states = {k:None for k in self.rnn_keys}
        for k in self.rnn_states:
            if 'phi' in k:
                self.rnn_states[k] = self.algorithm.model.network.phi_body.get_reset_states(cuda=self.algorithm.kwargs['use_cuda'])
            if 'critic' in k:
                self.rnn_states[k] = self.algorithm.model.network.critic_body.get_reset_states(cuda=self.algorithm.kwargs['use_cuda'])
            if 'actor' in k:
                self.rnn_states[k] = self.algorithm.model.network.actor_body.get_reset_states(cuda=self.algorithm.kwargs['use_cuda'])

    def _post_process(self, prediction):
        if self.recurrent:
            for k, (hs, cs) in self.rnn_states.items():
                for idx in range(len(hs)):
                    self.rnn_states[k][0][idx] = torch.Tensor(prediction['next_rnn_states'][k][0][idx].cpu())
                    self.rnn_states[k][1][idx] = torch.Tensor(prediction['next_rnn_states'][k][1][idx].cpu())
            
            for k, v in prediction.items():
                if isinstance(v, dict):
                    for vk, (hs,cs) in v.items():
                        for idx in range(len(hs)): 
                            hs[idx]=hs[idx].detach().cpu()
                            cs[idx]=cs[idx].detach().cpu()
                        prediction[k][vk] = (hs, cs)
                else:
                    prediction[k] = v.detach().cpu()
        else:
            prediction = {k: v.detach().cpu() for k, v in prediction.items()}

        return prediction

    def _pre_process_rnn_states(self, done=False):
        if done or self.rnn_states is None: self._reset_rnn_states()
        if self.algorithm.kwargs['use_cuda']:
            for k, (hs, cs) in self.rnn_states.items():
                for idx in range(len(hs)):
                    self.rnn_states[k][0][idx] = self.rnn_states[k][0][idx].cuda()
                    self.rnn_states[k][1][idx] = self.rnn_states[k][1][idx].cuda()
                
    def handle_experience(self, s, a, r, succ_s, done=False):
        non_terminal = torch.ones(1)*(1 - int(done))
        state = self.state_preprocessing(s)
        r = torch.ones(1)*r
        a = torch.from_numpy(a)

        self.algorithm.storage.add(self.current_prediction)
        self.algorithm.storage.add({'r': r, 'non_terminal': non_terminal, 's': state})

        self.handled_experiences += 1
        if self.training and self.handled_experiences >= self.algorithm.kwargs['horizon']:
            next_state = self.state_preprocessing(succ_s)
            
            if self.recurrent:
                self._pre_process_rnn_states(done=done)
                next_prediction = self.algorithm.model(next_state, rnn_states=self.rnn_states)
            else:
                next_prediction = self.algorithm.model(next_state)
            next_prediction = self._post_process(next_prediction)
            
            self.algorithm.storage.add(next_prediction)
            self.algorithm.train()
            self.handled_experiences = 0

    def take_action(self, state):
        state = self.state_preprocessing(state)
        
        if self.recurrent:
            self._pre_process_rnn_states()
            self.current_prediction = self.algorithm.model(state, rnn_states=self.rnn_states) 
        else:
            self.current_prediction = self.algorithm.model(state)
        self.current_prediction = self._post_process(self.current_prediction)
            
        return self.current_prediction['a'].numpy()

    def clone(self, training=None):
        clone = PPOAgent(name=self.name, algorithm=copy.deepcopy(self.algorithm))
        clone.training = training
        
        return clone


def build_PPO_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param config: Dict containing configuration for ppo agent
    :returns: PPOAgent adapted to be trained on :param: task under :param: config
    '''
    kwargs = config.copy()
    kwargs['state_preprocess'] = PreprocessFunction(task.observation_dim, kwargs['use_cuda'])

    input_dim = task.observation_dim
    if kwargs['phi_arch'] != 'None':
        output_dim = 64
        if kwargs['phi_arch'] == 'RNN':
            phi_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['phi_arch'] == 'MLP':
            phi_body = FCBody(input_dim, hidden_units=(output_dim,output_dim), gate=F.leaky_relu)
        input_dim = output_dim
    else :
        phi_body = None
    
    if kwargs['actor_arch'] != 'None':
        output_dim = 64
        if kwargs['actor_arch'] == 'RNN':
            actor_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['actor_arch'] == 'MLP':
            actor_body = FCBody(input_dim, hidden_units=(output_dim, output_dim), gate=F.leaky_relu)
    else :
        actor_body = None 
    
    if kwargs['critic_arch'] != 'None':
        output_dim = 64
        if kwargs['critic_arch'] == 'RNN':
            critic_body = LSTMBody(input_dim, hidden_units=(output_dim,), gate=F.leaky_relu)
        elif kwargs['critic_arch'] == 'MLP':
            critic_body = FCBody(input_dim, hidden_units=(output_dim, output_dim), gate=F.leaky_relu)
    else :
        critic_body = None 
    
    if task.action_type is 'Discrete' and task.observation_type is 'Discrete':
        model = CategoricalActorCriticNet(task.observation_dim, task.action_dim,
                                          phi_body=phi_body,
                                          actor_body=actor_body,
                                          critic_body=critic_body)
    if task.action_type is 'Continuous' and task.observation_type is 'Continuous':
        model = GaussianActorCriticNet(task.observation_dim, task.action_dim,
                                       phi_body=phi_body,
                                       actor_body=actor_body,
                                       critic_body=critic_body)

    model.share_memory()
    ppo_algorithm = PPOAlgorithm(kwargs, model)

    return PPOAgent(name=agent_name, algorithm=ppo_algorithm)
