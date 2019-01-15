import numpy as np
import random
import copy

import math
import os
import time 
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

from torch.multiprocessing import Process 

Transition = namedtuple('Transition', ('state','action','next_state', 'reward','done') )
TransitionPR = namedtuple('TransitionPR', ('idx','priority','state','action','next_state', 'reward','done') )

class ReplayBuffer(object) :
    def __init__(self,capacity) :
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) :
        if len(self.memory) < self.capacity :
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1) % self.capacity
        self.position = int(self.position)

    def sample(self,batch_size) :
        return random.sample(self.memory, batch_size)

    def __len__(self) :
        return len(self.memory)

EXP = namedtuple('EXP', ('state','action','next_state', 'reward','done') )
Transition = namedtuple('Transition', ('state','action','next_state', 'reward','done') )
TransitionPR = namedtuple('TransitionPR', ('idx','priority','state','action','next_state', 'reward','done') )


class PrioritizedReplayBuffer :
    def __init__(self,capacity, alpha=0.2) :
        self.length = 0
        self.counter = 0
        self.alpha = alpha
        self.epsilon = 1e-6
        self.capacity = int(capacity)
        self.tree = np.zeros(2*self.capacity-1)
        self.data = np.zeros(self.capacity,dtype=object)
        self.sumPi_alpha = 0.0
        
    def reset(self) :
        self.__init__(capacity=self.capacity,alpha=self.alpha)

    def add(self, exp, priority) :
        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity 
        
        idx = self.counter + self.capacity -1
        
        self.data[self.counter] = exp
        
        self.counter += 1
        self.length = min(self.length+1, self.capacity)
        if self.counter >= self.capacity :
            self.counter = 0
        
        self.sumPi_alpha += priority
        self.update(idx,priority)

    def priority(self, error) :
        return (error+self.epsilon)**self.alpha
            
    def update(self, idx, priority) :
        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity 
        
        change = priority - self.tree[idx]
        if change > 1e3 :
            print('BIG CHANGE HERE !!!!')
            print(change)
            raise Exception()

        previous_priority = self.tree[idx]
        self.sumPi_alpha -= previous_priority

        self.sumPi_alpha += priority
        self.tree[idx] = priority
        
        self._propagate(idx,change)

    def _propagate(self, idx, change) :
        parentidx = (idx - 1) // 2
        
        self.tree[parentidx] += change
        
        if parentidx != 0 :
            self._propagate(parentidx, change)
            
    def __call__(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1
        data = self.data[dataidx]
        priority = self.tree[idx]
        
        return (idx, priority, data)
    
    def get(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1
        
        data = self.data[dataidx]
        if not isinstance(data,EXP) :
            raise TypeError
                
        priority = self.tree[idx]
        
        return (idx, priority, *data)
    
    def get_importance_sampling_weight(priority,beta=1.0) :
        return pow( self.capacity * priority , -beta )

    def get_buffer(self) :
        return [ self.data[i] for i in range(self.capacity) if isinstance(self.data[i],EXP) ]

            
    def _retrieve(self,idx,s) :
         leftidx = 2*idx+1
         rightidx = leftidx+1
         
         if leftidx >= len(self.tree) :
            return idx
         
         if s <= self.tree[leftidx] :
            return self._retrieve(leftidx, s)
         else :
            return self._retrieve(rightidx, s-self.tree[leftidx])
            
    def total(self) :
        return self.tree[0]

    def __len__(self) :
        return self.length






def hard_update(fromm, to) :
    for fp, tp in zip( fromm.parameters(), to.parameters() ) :
        fp.data.copy_( tp.data )

def soft_update(fromm, to, tau) :
    for fp, tp in zip( fromm.parameters(), to.parameters() ) :
        fp.data.copy_( (1.0-tau)*fp.data + tau*tp.data ) 


def LeakyReLU(x) :
    return F.leaky_relu(x,0.1)


class DQN(nn.Module) :
    def __init__(self,nbr_actions=2,actfn=LeakyReLU, useCNN={'use':True,'dim':3}, use_cuda=False ) :
        super(DQN,self).__init__()
        
        self.nbr_actions = nbr_actions
        self.use_cuda = use_cuda

        self.actfn = actfn
        self.useCNN = useCNN

        """
        TODO :
        implement the cloning scheme for this:

        if self.useCNN['use'] :
            self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
            self.bn3 = nn.BatchNorm2d(32)
            #self.f = nn.Linear(192,128)
            self.f = nn.Linear(1120,32)
        
        else :
        """
        self.f1 = nn.Linear(self.useCNN['dim'], 1024)   
        self.f2 = nn.Linear(1024, 256)
        self.f = nn.Linear(256,64)  
        
        self.qsa = nn.Linear(64,self.nbr_actions)
        
        if self.use_cuda:
            self = self.cuda()
    

    def clone(self):
        cloned = DQN(nbr_actions=self.nbr_actions,actfn=self.actfn,useCNN=self.useCNN,use_cuda=self.use_cuda)
        cloned.load_state_dict( self.state_dict() )
        return cloned  

    def forward(self, x) :
        if self.useCNN['use'] :
            x = self.actfn( self.bn1(self.conv1(x) ) )
            x = self.actfn( self.bn2(self.conv2(x) ) )
            x = self.actfn( self.bn3(self.conv3(x) ) )
            x = x.view( x.size(0), -1)
        
            #print(x.size())
        else :
            x = self.actfn( self.f1(x) )
            x = self.actfn( self.f2(x) )

        fx = self.actfn( self.f(x) )

        qsa = self.qsa(fx)
        
        return qsa



class DuelingDQN(nn.Module) :
    def __init__(self,nbr_actions=2,actfn=LeakyReLU, useCNN={'use':True,'dim':3}, use_cuda=False ) :
        super(DuelingDQN,self).__init__()
        
        self.nbr_actions = nbr_actions
        self.use_cuda = use_cuda

        self.actfn = actfn
        self.useCNN = useCNN

        """
        TODO :
        implement the cloning scheme for this:

        if self.useCNN['use'] :
            self.conv1 = nn.Conv2d(3,16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
            self.bn3 = nn.BatchNorm2d(32)
            #self.f = nn.Linear(192,128)
            self.f = nn.Linear(1120,32)
        
        else :
        """
        self.f1 = nn.Linear(self.useCNN['dim'], 1024)   
        self.f2 = nn.Linear(1024, 256)
        self.f = nn.Linear(256,64)  
        
        self.value = nn.Linear(64,1)
        self.advantage = nn.Linear(64,self.nbr_actions)

        if self.use_cuda:
            self = self.cuda()
    
    def clone(self):
        cloned = DuelingDQN(nbr_actions=self.nbr_actions,actfn=self.actfn,useCNN=self.useCNN,use_cuda=self.use_cuda)
        cloned.load_state_dict( self.state_dict() )
        return cloned  
        
    def forward(self, x) :
        try:
            if self.useCNN['use'] :
                x = self.actfn( self.bn1(self.conv1(x) ) )
                x = self.actfn( self.bn2(self.conv2(x) ) )
                x = self.actfn( self.bn3(self.conv3(x) ) )
                x = x.view( x.size(0), -1)
            
                #print(x.size())
            else :
                x = self.actfn( self.f1(x) )
                x = self.actfn( self.f2(x) )
        except Exception as e:
            raise e 

        fx = self.actfn( self.f(x) )

        v = self.value(fx)
        va = torch.cat( [ v for i in range(self.nbr_actions) ], dim=1)
        a = self.advantage(fx)

        suma = torch.mean(a,dim=1,keepdim=True)
        suma = torch.cat( [ suma for i in range(self.nbr_actions) ], dim=1)
        
        x = va+a-suma
            
        return x

class DeepQNetworkAlgorithm :
    def __init__(self,kwargs) :
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
            "tau": float, soft-update rate [default: tau=1e-3].
            "gamma": float, Q-learning gamma rate [default: gamma=0.999].

            "preprocess": preprocessing function/transformation to apply to observations [default: preprocess=T.ToTensor()]
            
            "nbr_worker": int to specify whether to use the Distributed variant of DQN and how many worker to use [default: nbr_worker=1].
        """

        """
        TODO :
        So far, there is only one replay buffer that stores all the experiences,
        and there can be many learner/worker that samples from it asynchronously.

        Investigate the possibility of sampling asynchronously from multiple replay buffers
        that are filled up sequentially in the handle_experience function.
        It might yields better stability.
        It would be necessary to soft_update/hard_update the target models at a different rate than currently...

        """

        """
        TODO:
        So far, each worker/learner encompasses its own pair of target and working model.

        Investigate the use of only one target model that is converging towards the main model
        and shared among all the worker/learner threads.

        """
        
        self.kwargs = kwargs
        self.use_cuda = kwargs["use_cuda"]
        
        self.model = kwargs["model"]
        
        self.nbr_worker = kwargs["nbr_worker"]

        self.target_model = copy.deepcopy(self.model)
        hard_update(self.target_model,self.model)
        if self.use_cuda :
            self.target_model = self.target_model.cuda()

        if kwargs["use_PER"] :
            self.replayBuffer = PrioritizedReplayBuffer(capacity=kwargs["replay_capacity"],alpha=kwargs["PER_alpha"])
        else :
            self.replayBuffer = ReplayMemory(capacity=kwargs["replay_capacity"])
        
        self.min_capacity = kwargs["min_capacity"]
        self.batch_size = kwargs["batch_size"]

        self.lr = kwargs["lr"]
        self.TAU = kwargs["tau"]
        self.GAMMA = kwargs["gamma"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr )
        
        self.preprocess = kwargs["preprocess"]

        self.epsend = 0.05
        self.epsstart = 0.9
        self.epsdecay = 10

    def clone(self) :
        cloned_kwargs = self.kwargs
        cloned_model = self.model.clone()
        self.kwargs['model'] = cloned_model
        cloned = DeepQNetworkAlgorithm(kwargs=cloned_kwargs)
        
        # TODO : decide whether to transfer the replay buffer or not.
        #cloned.replayBuffer = self.replayBuffer
        
        return cloned
        
    def optimize_model(self) :
        """
        1) Estimate the gradients of the loss with respect to the
        current learner model on a batch of experiences sampled 
        from the Prioritized Experience Replay buffer.
        2) Backward the loss.
        3) Update the weights with the optimizer.
        4) Update the Prioritized Experience Replay buffer with new priorities.
        
        :returns loss_np: numpy scalar of the estimated loss function. 
        """
        
        if len(self.replayBuffer) < self.min_capacity :
            return None
        
        #Create batch with PrioritizedReplayBuffer/PER:
        prioritysum = self.replayBuffer.total()
        
        # Random Experience Sampling with priority
        #randexp = np.random.random(size=self.batch_size)*prioritysum
        
        # Sampling within each sub-interval :
        #step = prioritysum / self.batch_size
        #randexp = np.arange(0.0,prioritysum,step)
        fraction = 0.0#0.8
        low = 0.0#fraction*prioritysum 
        step = (prioritysum-low) / self.batch_size
        try:
            randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(self.batch_size))
        except Exception as e :
            print( prioritysum, step)
            raise e 
        # Sampling within each sub-interval with (un)trunc normal priority over the top :
        #randexp = np.random.normal(loc=0.75,scale=1.0,size=self.batch_size) * prioritysum

        
        batch = list()
        priorities = []
        for i in range(self.batch_size):
            try :
                el = self.replayBuffer.get(randexp[i])
                priorities.append( el[1] )
                batch.append(el)
            except TypeError as e :
                continue
                #print('REPLAY BUFFER EXCEPTION...')
        
        batch = TransitionPR( *zip(*batch) )
        
        # Create Batch with replayMemory :
        #transitions = replayBuffer.sample(self.batch_size)
        #batch = Transition(*zip(*transitions) )

        # Importance Sampling Weighting:
        beta = 1.0
        priorities = Variable( torch.from_numpy( np.array(priorities) ), requires_grad=False).float()
        importanceSamplingWeights = torch.pow( len(self.replayBuffer) * priorities , -beta)
        
        next_state_batch = Variable(torch.cat( batch.next_state), requires_grad=False)
        state_batch = Variable( torch.cat( batch.state) , requires_grad=False)
        action_batch = Variable( torch.cat( batch.action) , requires_grad=False)
        reward_batch = Variable( torch.cat( batch.reward ), requires_grad=False ).view((-1,1))
        done_batch = [ 0.0 if batch.done[i] else 1.0 for i in range(len(batch.done)) ]
        done_batch = Variable( torch.FloatTensor(done_batch), requires_grad=False ).view((-1,1))
        
        if self.use_cuda :
            importanceSamplingWeights = importanceSamplingWeights.cuda()
            next_state_batch = next_state_batch.cuda()
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            done_batch = done_batch.cuda()

        self.optimizer.zero_grad()

        state_action_values = self.model(state_batch)
        state_action_values_g = state_action_values.gather(dim=1, index=action_batch)
        
        ############################
        targetQ_nextS_A_values = self.target_model(next_state_batch)
        argmaxA_targetQ_nextS_A_values, index_argmaxA_targetQ_nextS_A_values = targetQ_nextS_A_values.max(1)
        argmaxA_targetQ_nextS_A_values = argmaxA_targetQ_nextS_A_values.view(-1,1)
        ############################

        # Compute the expected Q values
        gamma_next = (self.GAMMA * argmaxA_targetQ_nextS_A_values)#.type(torch.cuda.FloatTensor)
        expected_state_action_values = reward_batch + done_batch*gamma_next 
        
        # Compute loss:
        diff = expected_state_action_values - state_action_values_g
        is_diff_squared = importanceSamplingWeights.unsqueeze(1) * diff.pow(2.0)
        loss_per_item = is_diff_squared 
        loss = torch.mean( is_diff_squared)
        loss.backward()

        # TODO: 
        #
        # investigate clamping on the learner/worker's model gradients 
        # knowing that they are used as gradient accumulators...
        #
        """
        for param in model.parameters():
            if param.grad is not None :
                if param.grad.data is not None :
                    param.grad.data.clamp_(-1, 1)
        """

        
        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model.parameters()])
        weights_decay_loss.backward()
        
        self.optimizer.step()

        #UPDATE THE PR :
        loss_np = loss_per_item.cpu().data.numpy()
        for (idx, new_error) in zip(batch.idx,loss_np) :
            new_priority = self.replayBuffer.priority(new_error)
            self.replayBuffer.update(idx,new_priority)

        return loss_np

    def handle_experience(self, experience):
        '''
        This function is only called during training.
        It stores experience in the replay buffer.

        :param experience: EXP object containing the current, relevant experience.
        :returns:
        '''
        if self.kwargs["use_PER"] :
            init_sampling_priority =  self.replayBuffer.priority( torch.abs(experience.reward).cpu().numpy() )
            self.replayBuffer.add(experience,init_sampling_priority)
        else :
            self.replayBuffer.push( experience)

    def train(self,iteration=1) :
        for t in range(iteration) :
            lossnp = self.optimize_model()                   
            soft_update(self.target_model,self.model,self.TAU)    



class DoubleDeepQNetworkAlgorithm(DeepQNetworkAlgorithm) :

    def clone(self) :
        cloned_kwargs = self.kwargs
        cloned_model = self.model.clone()
        self.kwargs['model'] = cloned_model
        cloned = DoubleDeepQNetworkAlgorithm(kwargs=cloned_kwargs)
        
        # TODO : decide whether to transfer the replay buffer or not.
        #cloned.replayBuffer = self.replayBuffer
        
        return cloned
        

    def optimize_model(self) :
        """
        1) Estimate the gradients of the loss with respect to the
        current learner model on a batch of experiences sampled 
        from the Prioritized Experience Replay buffer.
        2) Backward the loss.
        3) Update the weights with the optimizer.
        4) Update the Prioritized Experience Replay buffer with new priorities.
        
        :returns loss_np: numpy scalar of the estimated loss function. 
        """
        
        if len(self.replayBuffer) < self.min_capacity :
            return None 
        
        #Create batch with PrioritizedReplayBuffer/PER:
        prioritysum = self.replayBuffer.total()
        
        # Random Experience Sampling with priority
        #randexp = np.random.random(size=self.batch_size)*prioritysum
        
        # Sampling within each sub-interval :
        #step = prioritysum / self.batch_size
        #randexp = np.arange(0.0,prioritysum,step)
        fraction = 0.0#0.8
        low = 0.0#fraction*prioritysum 
        step = (prioritysum-low) / self.batch_size
        try:
            randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(self.batch_size))
        except Exception as e :
            print( prioritysum, step)
            raise e 
        # Sampling within each sub-interval with (un)trunc normal priority over the top :
        #randexp = np.random.normal(loc=0.75,scale=1.0,size=self.batch_size) * prioritysum

        
        batch = list()
        priorities = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            try :
                el = self.replayBuffer.get(randexp[i])
                priorities[i] = el[1]
                batch.append(el)
            except TypeError as e :
                continue
                #print('REPLAY BUFFER EXCEPTION...')
        
        batch = TransitionPR( *zip(*batch) )
        
        # Create Batch with replayMemory :
        #transitions = replayBuffer.sample(self.batch_size)
        #batch = Transition(*zip(*transitions) )

        # Importance Sampling Weighting:
        beta = 1.0
        priorities = Variable( torch.from_numpy(priorities ), requires_grad=False).float()
        importanceSamplingWeights = torch.pow( len(self.replayBuffer) * priorities , -beta)
        
        next_state_batch = Variable(torch.cat( batch.next_state), requires_grad=False)
        state_batch = Variable( torch.cat( batch.state) , requires_grad=False)
        action_batch = Variable( torch.cat( batch.action) , requires_grad=False)
        reward_batch = Variable( torch.cat( batch.reward ), requires_grad=False ).view((-1,1))
        done_batch = [ 0.0 if batch.done[i] else 1.0 for i in range(len(batch.done)) ]
        done_batch = Variable( torch.FloatTensor(done_batch), requires_grad=False ).view((-1,1))
        
        if self.use_cuda :
            importanceSamplingWeights = importanceSamplingWeights.cuda()
            next_state_batch = next_state_batch.cuda()
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            done_batch = done_batch.cuda()

        self.optimizer.zero_grad()

        state_action_values = self.model(state_batch)
        state_action_values_g = state_action_values.gather(dim=1, index=action_batch)

        ############################
        targetQ_nextS_A_values = self.target_model(next_state_batch)
        Q_nextS_A_values = self.model(next_state_batch)
        index_argmaxA_Q_nextS_A_values = Q_nextS_A_values.max(1)[1].view(-1,1)
        targetQ_nextS_argmaxA_Q_nextS_A_values = targetQ_nextS_A_values.gather( dim=1, index=index_argmaxA_Q_nextS_A_values).detach().view((-1,1))
        ############################

        # Compute the expected Q values
        gamma_next = (self.GAMMA * targetQ_nextS_argmaxA_Q_nextS_A_values).type(FloatTensor)
        expected_state_action_values = reward_batch + done_batch*gamma_next 

        # Compute loss:
        diff = expected_state_action_values - state_action_values_g
        is_diff_squared = importanceSamplingWeights * diff.pow(2.0)
        loss = torch.mean( is_diff_squared)
        loss.backward()
        
        # TODO: 
        #
        # investigate clamping on the learner/worker's model gradients 
        # knowing that they are used as gradient accumulators...
        #
        """
        for param in model.parameters():
            if param.grad is not None :
                if param.grad.data is not None :
                    param.grad.data.clamp_(-1, 1)
        """
        
        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model.parameters()])
        weights_decay_loss.backward()
        
        self.optimizer.step()
        
        #UPDATE THE PR :
        loss_np = loss.cpu().data.numpy()
        for (idx, new_error) in zip(batch.idx,loss_np) :
            new_priority = self.replayBuffer.priority(new_error)
            self.replayBuffer.update(idx,new_priority)

        return loss_np

class DeepQNetworkAgent2Queue():
    def __init__(self, dqnAgent, training=False):
        self.name = dqnAgent.name 
        self.training = training
        
        if isinstance(dqnAgent,DeepQNetworkAgent):                
            self.kwargs = dict()
            for name in dqnAgent.kwargs:
                if 'model' in name :
                    continue
                else :
                    self.kwargs[name] = dqnAgent.kwargs[name]
            
            self.kwargs['model'] = dqnAgent.kwargs["model"].state_dict()
            for name in self.kwargs["model"] :
                self.kwargs["model"][name] = self.kwargs["model"][name].cpu().numpy()
        else :
            # cloning :
            self.kwargs = copy.deepcopy(dqnAgent.kwargs) 

    def queue2policy(self):
        for name in self.kwargs["model"] :
            self.kwargs["model"][name] = torch.from_numpy( self.kwargs["model"][name] )
        
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


class DeepQNetworkAgent():
    def __init__(self, network, algorithm):
        """
        :param network: model network to be optimized by the algorithm.
        :param algorithm: algorithm class to use to optimize the network.
        """

        #self.network = network
        self.algorithm = algorithm
        self.training = False
        self.hashing_function = self.algorithm.kwargs["preprocess"]
        
        self.kwargs = algorithm.kwargs

        self.epsend = self.kwargs['epsend']
        self.epsstart = self.kwargs['epsstart']
        self.epsdecay = self.kwargs['epsdecay']
        self.nbr_steps = 0 

        self.name = self.kwargs['name']

    def handle_experience(self, s, a, r, succ_s,done=False):
        hs = self.hashing_function(s)
        hsucc = self.hashing_function(succ_s)
        r = torch.ones(1)*r
        a = torch.from_numpy(a)
        experience = EXP(hs,a, hsucc,r,done)
        self.algorithm.handle_experience(experience=experience)
        
        if self.training :
            self.algorithm.train(iteration=1)
        
    def take_action(self, state):
        self.nbr_steps += 1
        self.eps = self.epsend + (self.epsstart-self.epsend) * math.exp(-1.0 * self.nbr_steps / self.epsdecay )
        
        action,qsa = self.select_action(model=self.algorithm.model,state=self.hashing_function(state),eps=self.eps)
        
        return action

    def reset_eps(self):
        self.eps = self.epsstart
        
    def select_action(self,model,state,eps) :
        sample = random.random()
        if sample > eps :
            output = model( Variable(state) ).cpu().data
            qsa, action = output.max(1)
            action = action.view(1,1)
            qsa = output.max(1)[0].view(1,1)[0,0]
            return action.numpy(), qsa
        else :
            random_action = torch.LongTensor( [[random.randrange(self.algorithm.model.nbr_actions) ] ] )
            return random_action.numpy(), 0.0


    def clone4queue(self,training=False) :
        policy2queue = DeepQNetworkAgent2Queue(self,training=training)
        return policy2queue

    def clone(self, training=False):
        """
        TODO : decide whether to launch the training automatically or do it manually.
        So far it is being done manually...
        """

        #cloned_network = self.network.clone()
        cloned_algorithm = self.algorithm.clone()
        cloned = DeepQNetworkAgent(network=None, algorithm=cloned_algorithm)
        cloned.reset_eps()
        
        cloned.training = training 

        return cloned

class PreprocessFunction(object) :
    def __init__(self, hash_function, state_space_size,use_cuda):
        self.hash_function = hash_function
        self.state_space_size = state_space_size
        self.use_cuda = use_cuda
    def __call__(self,x) :
        x = self.hash_function(x)
        one_hot_encoded_state = np.zeros(self.state_space_size)
        one_hot_encoded_state[x] = 1.0
        if self.use_cuda :
            return torch.from_numpy( one_hot_encoded_state ).unsqueeze(0).type(torch.cuda.FloatTensor)
        else :
            return torch.from_numpy( one_hot_encoded_state ).unsqueeze(0).type(torch.FloatTensor)
            
def build_DQN_Agent(state_space_size=32, 
                        action_space_size=3, 
                        hash_function=None,
                        double=False,
                        dueling=False, 
                        num_worker=1, 
                        use_PER=True,
                        alphaPER = 0.8,
                        MIN_MEMORY = 1e1,
                        epsstart=0.9,
                        epsend=0.05,
                        epsdecay=1e3,
                        use_cuda = False
                        ):
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
        "tau": float, soft-update rate [default: tau=1e-3].
        "gamma": float, Q-learning gamma rate [default: gamma=0.999].

        "preprocess": preprocessing function/transformation to apply to observations [default: preprocess=T.ToTensor()]
        
        "worker_nbr_steps_max": int, number of steps of the training loop for each worker/learner [default: worker_nbr_steps_max=1000].

        "nbr_worker": int to specify whether to use the Distributed variant of DQN and how many worker to use [default: nbr_worker=1].

        "epsstart":
        "epsend":
        "epsdecay":

        "dueling":
        "double":
        "nbr_actions":
        "actfn":
        "useCNN":
    """

    """
    TODO : implement CNN usage for DQN...
    """
    useCNN = {'use':False,'dim':state_space_size}
    if useCNN['use']:
        preprocess_model = T.Compose([T.ToPILImage(),
                    T.Scale(64, interpolation=Image.CUBIC),
                    T.ToTensor() ] )
    else :
        preprocess_model = T.Compose([
                    T.ToTensor() ] )

    if hash_function is not None :
        kwargs['hash_function'] = hash_function
        preprocess = PreprocessFunction(hash_function=hash_function, state_space_size=state_space_size,use_cuda=use_cuda)
    else :
        """
        TODO :
        """
        preprocess = (lambda x: preprocess_model(x))

    kwargs["worker_nbr_steps_max"] = 10
    kwargs["nbr_actions"] = action_space_size
    kwargs["actfn"] = LeakyReLU
    kwargs["useCNN"] = useCNN
    # Create model architecture:
    if dueling :
        model = DuelingDQN(action_space_size,useCNN=useCNN, use_cuda=use_cuda)
        print("Dueling DQN model initialized: OK")
    else :
        model = DQN(action_space_size,useCNN=useCNN, use_cuda=use_cuda)
        print("DQN model initialized: OK")
    model.share_memory()
    
    kwargs["model"] = model
    kwargs["dueling"] = dueling
    kwargs["double"] = double 

    BATCH_SIZE = 32#256
    GAMMA = 0.99
    TAU = 1e-3
    lr = 1e-3
    memoryCapacity = 25e3
    
    name = "DQN"
    if dueling : name = 'Dueling'+name 
    if double : name = 'Double'+name 
    name += '+GAMMA{}+TAU{}'.format(GAMMA,TAU)+'+PER-alpha'+str(alphaPER)+'-w'+str(num_worker)+'-lr'+str(lr)+'-b'+str(BATCH_SIZE)+'-m'+str(memoryCapacity)
    
    model_path = './'+name 
    
    path=model_path

    kwargs['name'] = name 
    kwargs["path"] = path 
    kwargs["use_cuda"] = use_cuda 

    # Initialize replay buffer:
    kwargs["replay_capacity"] = memoryCapacity
    kwargs["min_capacity"] = MIN_MEMORY
    kwargs["batch_size"] = BATCH_SIZE
    kwargs["use_PER"] = use_PER
    kwargs["PER_alpha"] = alphaPER

    kwargs["lr"] = lr 
    kwargs["tau"] = TAU 
    kwargs["gamma"] = GAMMA

    kwargs["preprocess"] = preprocess
    kwargs["nbr_worker"] = num_worker

    kwargs['epsstart'] = epsstart
    kwargs['epsend'] = epsend
    kwargs['epsdecay'] = epsdecay
    

    if dueling :
        DeepQNetwork_algo = DoubleDeepQNetworkAlgorithm(kwargs=kwargs)
    else :
        DeepQNetwork_algo = DeepQNetworkAlgorithm(kwargs=kwargs)
    
    agent = DeepQNetworkAgent(network=model,algorithm=DeepQNetwork_algo)
    
    return agent

if __name__ == "__main__":
    build_DQN_Agent(double=True,dueling=True)