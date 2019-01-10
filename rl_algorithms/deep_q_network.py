import numpy as np
import random
import copy

import math
import os
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T

"""
from threading import Thread, Lock, Event
"""
from torch.multiprocessing import Process, Lock, Event 

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
        self.mutex = Lock()
        self.update_mutex = Lock()

    def lock(self) :
        self.mutex.acquire()

    def unlock(self) :
        self.mutex.release()

    def update_lock(self) :
        self.update_mutex.acquire()

    def update_unlock(self) :
        self.update_mutex.release()

    def reset(self) :
        self.__init__(capacity=self.capacity,alpha=self.alpha)

    def add(self, exp, priority) :
        self.lock()

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

        self.unlock()

    def priority(self, error) :
        return (error+self.epsilon)**self.alpha
            
    def update(self, idx, priority) :
        self.update_lock()

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
        
        self.update_unlock()

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



class DQN(nn.Module) :
    def __init__(self,nbr_actions=2,actfn= lambda x : F.leaky_relu(x,0.1), useCNN={'use':True,'dim':3}, use_cuda=True ) :
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
        
        self.mutex = Lock()

    def clone(self):
        cloned = DQN(nbr_actions=self.nbr_actions, actfn=self.actfn, useCNN=self.useCNN, use_cuda=self.use_cuda)

        cloned.f1 = copy.deepcopy(self.f1)
        cloned.f2 = copy.deepcopy(self.f2)
        cloned.f = copy.deepcopy(self.f)
        cloned.qsa = copy.deepcopy(self.qsa)
        
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

    
    def lock(self) :
        self.mutex.acquire()

    def unlock(self) :
        self.mutex.release()

def LeakyReLU(x) :
    return F.leaky_relu(x,0.1)

class DuelingDQN(nn.Module) :
    def __init__(self,nbr_actions=2,actfn=LeakyReLU, useCNN={'use':True,'dim':3}, use_cuda=True ) :
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
        
        self.mutex = Lock()

    def clone(self):
        cloned = DuelingDQN(nbr_actions=self.nbr_actions, actfn=self.actfn, useCNN=self.useCNN, use_cuda=self.use_cuda)

        cloned.f1 = copy.deepcopy(self.f1)
        cloned.f2 = copy.deepcopy(self.f2)
        cloned.f = copy.deepcopy(self.f)
        cloned.value = copy.deepcopy(self.value)
        cloned.advantage = copy.deepcopy(self.advantage)

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

    
    def lock(self) :
        self.mutex.acquire()

    def unlock(self) :
        self.mutex.release()




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
            
            "w2m_update_interval": int, worker2model update interval used for each worker/learner [default: w2m_update_interval=10].

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
        self.model.eval()

        self.nbr_worker = kwargs["nbr_worker"]

        self.wmodels = []
        self.paths = []
        for i in range(self.nbr_worker) :
            self.wmodels.append( self.model.clone() )
            self.wmodels[-1].training = True
            hard_update(self.wmodels[-1],self.model)
            if self.use_cuda :
                self.wmodels[-1] = self.wmodels[-1].cuda()
            
            self.paths.append( kwargs["path"]+str(i) )
        

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

        
        self.worker_nbr_steps_max = kwargs["worker_nbr_steps_max"]
        self.w2m_update_interval = kwargs["w2m_update_interval"]
        self.epsend = 0.05
        self.epsstart = 0.9
        self.epsdecay = 10

        self.workerfns = []
        self.processes = []
        self.stops = []
        for i in range(self.nbr_worker) :
            self.stops.append(Event())
            """
            self.workerfns.append( lambda: self.train(
                                        worker_index=i,
                                        model=self.wmodels[i],
                                        replayBuffer=self.replayBuffer,
                                        optimizer=self.optimizer,
                                        stop=self.stops[-1],
                                        epsend=self.epsend,
                                        epsstart=self.epsstart,
                                        epsdecay=self.epsdecay
                                        )
            )

            #self.processes.append( Process(target=self.workerfns[i], name='DQN_Proc_{}'.format(i)) )
            self.processes.append( Thread(target=self.workerfns[i], name='DQN_Proc_{}'.format(i)) )
            """
            #self.processes.append( Thread(target=self.train, name='DQN_Proc_{}'.format(i),
            self.processes.append( Process(target=self.train, name='DQN_Proc_{}'.format(i),
                                            args=(i,
                                                self.wmodels[i],
                                                self.replayBuffer,
                                                self.optimizer,
                                                self.stops[-1],
                                                self.epsend,
                                                self.epsstart,
                                                self.epsdecay,
                                                )
                                            )
            )

    def clone(self) :
        """
        Clone this Algorithm instance. 
        Whatever the state of this instance, the cloned instance 
        is in a state ready to be launched on a training task
        using the start_all method.
        """
        cloned = DeepQNetworkAlgorithm(kwargs=self.kwargs)
        return cloned

    def start_all(self) :
        for i in range(self.nbr_worker):
            self.start(index=i)

    def start(self, index=0) :
        self.processes[index].start()

    def join_all(self) :
        for i in range(self.nbr_worker):
            self.join(index=i)
    
    def join(self, index=0) :
        self.processes[index].join()
    
    def stop_all(self) :
        for i in range(self.nbr_worker):
            self.stop(index=i)

    def stop(self, index=0) :
        self.stops[index].set()
        #self.stops[index] = True 

    
    def from_worker2model(self, index=0) :
        self.model.lock()
        
        self.optimizer.zero_grad()

        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model.parameters()])
        weights_decay_loss.backward()
        
        for wparam, mparam in zip(self.wmodels[index].parameters(), self.model.parameters() ) :
            if mparam.grad is not None:
                if wparam.grad is not None :
                    mparam.grad =  mparam.grad + wparam.grad
                
        self.optimizer.step()

        #update wmodels to the current state of the  :
        for i in range(self.nbr_worker):
            hard_update(self.wmodels[i],self.model)

        #zero the working model gradients :
        self.wmodels[index].zero_grad()
        
        self.model.unlock()


    def optimize_model(self,model,target_model,replayBuffer) :
        """
        1) Estimate the gradients of the loss with respect to the
        current learner model on a batch of experiences sampled 
        from the Prioritized Experience Replay buffer.
        2) Accumulate the gradients in the learner model's grad container.
        3) Update the Prioritized Experience Replay buffer with new priorities.

        The main model's weights are updated later by the a call to the from_worker2model function
        and each worker/learner model's grad containers are used to update the main model's weighgts. 

        :param model: model with respect to which the loss is being optimized.
        :param target_model: target model used to evaluate the action-value in a Double DQN scheme.
        :param replayBuffer: Prioritized Experience Replay buffer from which the batch is sampled.
        :returns loss_np: numpy scalar of the estimated loss function. 
        """
        
        if len(replayBuffer) < self.min_capacity :
            return None
        
        #Create batch with PrioritizedReplayBuffer/PER:
        prioritysum = replayBuffer.total()
        
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
                el = replayBuffer.get(randexp[i])
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
        importanceSamplingWeights = torch.pow( len(replayBuffer) * priorities , -beta)
        
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

        state_action_values = model(state_batch)
        state_action_values_g = state_action_values.gather(dim=1, index=action_batch)

        ############################
        targetQ_nextS_A_values = target_model(next_state_batch)
        argmaxA_targetQ_nextS_A_values, index_argmaxA_targetQ_nextS_A_values = targetQ_nextS_A_values.max(1)
        argmaxA_targetQ_nextS_A_values = argmaxA_targetQ_nextS_A_values.view(-1,1)
        ############################

        # Compute the expected Q values
        gamma_next = (self.GAMMA * argmaxA_targetQ_nextS_A_values).type(FloatTensor)
        expected_state_action_values = reward_batch + done_batch*gamma_next 

        # Compute loss:
        diff = expected_state_action_values - state_action_values_g
        is_diff_squared = importanceSamplingWeights * diff.pow(2.0)
        loss = torch.mean( is_diff_squared)
        
        
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
        
        #UPDATE THE PR :
        loss_np = loss.cpu().data.numpy()
        for (idx, new_error) in zip(batch.idx,loss_np) :
            new_priority = replayBuffer.priority(new_error)
            replayBuffer.update(idx,new_priority)

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


    def train(self,worker_index,model,replayBuffer,optimizer,stop,epsend=0.05,epsstart=0.9,epsdecay=10): 
        print("LAUNCHING TRAINING on worker:{} : OK.".format(worker_index))
        try :
            target_model = model.clone()
            hard_update(target_model,model)
            target_model.eval()
            if self.use_cuda :
                target_model = target_model.cuda()
                
            for t in count() :
                #lossnp = self.optimize_model(model,target_model,replayBuffer,optimizer)                   
                #print("WORKER:{} : optimizing : ...".format(worker_index))
                lossnp = self.optimize_model(model,target_model,replayBuffer)                   
                #print("WORKER:{} : optimizing : OK.".format(worker_index))
                
                # SOFT UPDATE :
                #print("WORKER:{} : soft update : ...".format(worker_index))
                soft_update(target_model,model,self.TAU)
                #print("WORKER:{} : soft update : OK.".format(worker_index))
                
                if t % self.w2m_update_interval == 0:
                    #print("WORKER:{} : learner sync : ...".format(worker_index))
                    self.from_worker2model()
                    #print("WORKER:{} : learner sync : OK.".format(worker_index))
                    
                #if self.stops[worker_index]:
                if stop.is_set() :#or t>self.worker_nbr_steps_max:
                    print("STOPPING TRAINING on worker:{} : ...".format(worker_index))
                    break

        except Exception as e :
            #bashlogger.exception(e)
            print(e)
            raise e 


        print("STOPPING TRAINING on worker:{} : OK.".format(worker_index))


class DoubleDeepQNetworkAlgorithm(DeepQNetworkAlgorithm) :

    def clone(self) :
        """
        Clone this Algorithm instance. 
        Whatever the state of this instance, the cloned instance 
        is in a state ready to be launched on a training task
        using the start_all method.
        """
        cloned = DoubleDeepQNetworkAlgorithm(kwargs=self.kwargs)
        return cloned
        

    def optimize_model(self,model,target_model,replayBuffer) :
        """
        1) Estimate the gradients of the loss with respect to the
        current learner model on a batch of experiences sampled 
        from the Prioritized Experience Replay buffer.
        2) Accumulate the gradients in the learner model's grad container.
        3) Update the Prioritized Experience Replay buffer with new priorities.

        The main model's weights are updated later by the a call to the from_worker2model function
        and each worker/learner model's grad containers are used to update the main model's weighgts. 

        :param model: model with respect to which the loss is being optimized.
        :param target_model: target model used to evaluate the action-value in a Double DQN scheme.
        :param replayBuffer: Prioritized Experience Replay buffer from which the batch is sampled.
        :returns loss_np: numpy scalar of the estimated loss function. 
        """
        
        if len(replayBuffer) < self.min_capacity :
            return None 
        
        #Create batch with PrioritizedReplayBuffer/PER:
        prioritysum = replayBuffer.total()
        
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
                el = replayBuffer.get(randexp[i])
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
        importanceSamplingWeights = torch.pow( len(replayBuffer) * priorities , -beta)
        
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

        state_action_values = model(state_batch)
        state_action_values_g = state_action_values.gather(dim=1, index=action_batch)

        ############################
        targetQ_nextS_A_values = target_model(next_state_batch)
        Q_nextS_A_values = model(next_state_batch)
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
        
        #UPDATE THE PR :
        loss_np = loss.cpu().data.numpy()
        for (idx, new_error) in zip(batch.idx,loss_np) :
            new_priority = replayBuffer.priority(new_error)
            replayBuffer.update(idx,new_priority)

        return loss_np

class DeepQNetworkAgent():
    def __init__(self, network, algorithm):
        """
        :param network: model network to be optimized by the algorithm.
        :param algorithm: algorithm class to use to optimize the network.
        """

        self.network = network
        self.algorithm = algorithm
        self.training = False
        self.hashing_function = self.algorithm.kwargs["preprocess"]
        
        self.kwargs = algorithm.kwargs

        self.epsend = self.kwargs['epsend']
        self.epsstart = self.kwargs['epsstart']
        self.epsdecay = self.kwargs['epsdecay']
        self.nbr_steps = 0 

        self.name = self.kwargs['name']

        self.mutex = Lock()

    def lock(self) :
        self.mutex.acquire()

    def unlock(self) :
        self.mutex.release()

    def launch_training(self):
        print("Launching training: ...")
        self.algorithm.start_all()
        self.training = True 
        print("Launching training: OK.")

    def stop_training(self):
        print("Stopping training: ...")
        self.algorithm.stop_all()
        self.training = False
        print("Stopping training: OK.")

    def handle_experience(self, s, a, r, succ_s,done=False):
        self.lock()
        hs = self.hashing_function(s)
        hsucc = self.hashing_function(succ_s)
        r = torch.ones(1)*r
        a = torch.from_numpy(a)
        experience = EXP(hs,a, hsucc,r,done)
        self.algorithm.handle_experience(experience=experience)
        self.unlock()

    def take_action(self, state):
        self.lock()
        self.nbr_steps += 1
        self.eps = self.epsend + (self.epsstart-self.epsend) * math.exp(-1.0 * self.nbr_steps / self.epsdecay )
        
        action,qsa = self.select_action(model=self.network,state=self.hashing_function(state),eps=self.eps)
        
        self.unlock()
        return action

    def reset_eps():
        self.lock()
        self.eps = self.epsstart
        self.unlock()

    def select_action(self,model,state,eps) :
        sample = random.random()
        if sample > eps :
            output = model( Variable(state) ).cpu().data
            qsa, action = output.max(1)
            action = action.view(1,1)
            qsa = output.max(1)[0].view(1,1)[0,0]
            return action.numpy(), qsa
        else :
            random_action = torch.LongTensor( [[random.randrange(self.network.nbr_actions) ] ] )
            return random_action.numpy(), 0.0

    def clone(self, training=False):
        """
        TODO : decide whether to launch the training automatically or do it manually.
        So far it is being done manually...
        """

        cloned_network = self.network.clone()
        cloned_algorithm = self.algorithm.clone()
        cloned = DeepQNetworkAgent(network=cloned_network, algorithm=cloned_algorithm)
        
        #if training :
        #    cloned.launch_training()

        return cloned

class PreprocessFunction(object) :
    def __init__(self, hash_function):
        self.hash_function = hash_function

    def __call__(self,x) :
        return torch.from_numpy( self.hash_function(x) ).unsqueeze(0).type(torch.cuda.FloatTensor)

def build_DQN_Agent(state_space_size=32, 
                        action_space_size=3, 
                        hash_function=None,
                        double=False,
                        dueling=False, 
                        num_worker=1, 
                        MIN_MEMORY = 1e1,
                        epsstart=0.9,
                        epsend=0.05,
                        epsdecay=1e3
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
        
        "w2m_update_interval": int, worker2model update interval used for each worker/learner [default: w2m_update_interval=10].
        "worker_nbr_steps_max": int, number of steps of the training loop for each worker/learner [default: worker_nbr_steps_max=1000].

        "nbr_worker": int to specify whether to use the Distributed variant of DQN and how many worker to use [default: nbr_worker=1].

        "epsstart":
        "epsend":
        "epsdecay":
    """

    use_cuda = True 

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
        if use_cuda :
            kwargs['hash_function'] = hash_function
            preprocess = PreprocessFunction(hash_function=hash_function)#(lambda x: torch.from_numpy( hash_function(x) ).unsqueeze(0).type(torch.cuda.FloatTensor))
        else :
            preprocess = (lambda x: torch.from_numpy(np.array( hash_function(x) )).unsqueeze(0).type(torch.FloatTensor))
        #preprocess = (lambda x: torch.from_numpy( np.ones((1,1))*hash_function(x)).type(torch.cuda.FloatTensor) )
        #preprocess = (lambda x: preprocess_model(hash_function(x)))
    else :
        """
        TODO :
        """
        preprocess = (lambda x: preprocess_model(x))

    kwargs["worker_nbr_steps_max"] = 10

    # Create model architecture:
    if dueling :
        model = DuelingDQN(action_space_size,useCNN=useCNN, use_cuda=use_cuda)
        print("Dueling DQN model initialized: OK")
    else :
        model = DQN(action_space_size,useCNN=useCNN, use_cuda=use_cuda)
        print("DQN model initialized: OK")
    model.share_memory()
    
    kwargs["model"] = model

    numep = 1000
    BATCH_SIZE = 256
    GAMMA = 0.999
    TAU = 1e-2
    alphaPER = 0.8
    lr = 1e-3
    memoryCapacity = 25e3
    
    name = 'CNN+DuelingDoubleDQN+WithZG+GAMMA{}+TAU{}'.format(GAMMA,TAU)\
    +'+IS+PER-alpha'+str(alphaPER) \
    +'-w'+str(num_worker)+'-lr'+str(lr)+'-b'+str(BATCH_SIZE)+'-m'+str(memoryCapacity)+'/'

    model_path = './'+name 
    
    path=model_path

    kwargs['name'] = name 
    kwargs["path"] = path 
    kwargs["use_cuda"] = True 

    # Initialize replay buffer:
    #memory = PrioritizedReplayBuffer(capacity=replay_capacity,alpha=PER_alpha)
    kwargs["replay_capacity"] = memoryCapacity
    kwargs["min_capacity"] = MIN_MEMORY
    kwargs["batch_size"] = BATCH_SIZE
    kwargs["use_PER"] = True
    kwargs["PER_alpha"] = alphaPER

    kwargs["lr"] = lr 
    kwargs["tau"] = TAU 
    kwargs["gamma"] = GAMMA

    kwargs["preprocess"] = preprocess
    kwargs["w2m_update_interval"] = 10
    kwargs["nbr_worker"] = num_worker

    kwargs['epsstart'] = epsstart
    kwargs['epsend'] = epsend
    kwargs['epsdecay'] = epsdecay
    

    DeepQNetwork_algo = DeepQNetworkAlgorithm(kwargs=kwargs)
    print("DeepQNetworkAlgorithm initialized: OK")

    agent = DeepQNetworkAgent(network=model,algorithm=DeepQNetwork_algo)
    print("DQN agent initialized: OK")

    return agent

def run():
    use_cuda = True
    rendering = False
    MAX_STEPS = 1000
    REWARD_SCALER = 1.0

    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

    def get_screen(task,action,preprocess) :
        global REWARD_SCALER
        screen, reward, done, info = task.step(action)
        reward = reward/REWARD_SCALER
        #screen = screen.transpose( (2,0,1) )
        #screen = np.ascontiguousarray( screen, dtype=np.float32) / 255.0
        screen = np.ascontiguousarray( screen, dtype=np.float32)
        screen = torch.from_numpy(screen)
        #screen = preprocess(screen)
        screen = screen.unsqueeze(0)
        #screen = screen.type(Tensor)
        return screen, reward, done, info

    def get_screen_reset(task,preprocess) :
        screen = task.reset()
        #screen = screen.transpose( (2,0,1) )
        #screen = np.ascontiguousarray( screen, dtype=np.float32) / 255.0
        screen = np.ascontiguousarray( screen, dtype=np.float32)
        screen = torch.from_numpy(screen)
        #screen = preprocess(screen)
        screen = screen.unsqueeze(0)
        return screen


    def select_action(model,state,steps_done=[],epsend=0.05,epsstart=0.9,epsdecay=200) :
        global nbr_actions
        sample = random.random()
        if steps_done is [] :
            steps_done.append(0)

        eps_threshold = epsend + (epsstart-epsend) * math.exp(-1.0 * steps_done[0] / epsdecay )
        steps_done[0] +=1

        #print('SAMPLE : {} // EPS THRESH : {}'.format(sample, eps_threshold) )
        if sample > eps_threshold :
            output = model( Variable(state, volatile=True).type(FloatTensor) ).data
            action = output.max(1)[1].view(1,1)
            qsa = output.max(1)[0].view(1,1)[0,0]
            return action, qsa
        else :
            return LongTensor( [[random.randrange(nbr_actions) ] ] ), 0.0

    def exploitation(model,state) :
        global nbr_actions
        output = model( Variable(state, volatile=True).type(FloatTensor) ).data.max(1)
        action = output[1].view(1,1)
        qsa = output[0].view(1,1)[0,0]
        return action,qsa


if __name__ == "__main__":
    build_DQN_Agent()
    build_DQN_Agent(double=True,dueling=True)