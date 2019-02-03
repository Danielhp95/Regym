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

EPS = 3e-1

EXP = namedtuple('EXP', ('state','action','next_state', 'reward','done') )
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

    def save(self,path):
        path += '.rb'
        np.savez(path, memory=self.memory, position=np.asarray(self.position) )

    def load(self,path):
        path += '.rb.npz'
        data= np.load(path)
        self.memory =data['memory']
        self.position = int(data['position'])

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

    def save(self,path):
        path += '.prb'
        np.savez(path, tree=self.tree, data=self.data,
            length=np.asarray(self.length), sumPi=np.asarray(self.sumPi_alpha),
            counter=np.asarray(self.counter), alpha=np.asarray(self.alpha) )

    def load(self,path):
        path += '.prb.npz'
        data= np.load(path)
        self.tree =data['tree']
        self.data = data['data']
        self.counter = int(data['counter'])
        self.length = int(data['length'])
        self.sumPi_alpha = float(data['sumPi'])
        self.alpha = float(data['alpha'])

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


def init_weights(size):
    v = 1. / np.sqrt(size[0])
    return torch.Tensor(size).uniform_(-v, v)

class ActorNN(nn.Module) :
    def __init__(self,state_dim=3,action_dim=2,action_scaler=1.0,useCNN={'use_cnn':False,'input_size':3},HER=False,actfn=LeakyReLU, use_cuda=False ) :
        super(ActorNN,self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scaler = action_scaler

        self.CNN = useCNN
        # dictionnary with :
        # - 'input_size' : int
        # - 'use_cnn' : bool
        if self.CNN['use_cnn'] :
            self.state_dim = self.CNN['input_size']

        self.HER = HER
        if self.HER :
            self.state_dim *= 2

        self.actfn = actfn
        #Features :
        '''
        TODO :
        implement the cloning scheme for this:

        if self.CNN['use_cnn'] :
            self.conv1 = nn.Conv2d(self.state_dim,16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
            self.bn3 = nn.BatchNorm2d(32)
            #self.featx = nn.Linear(448,self.nbr_actions)
            #self.featx = nn.Linear(192,128)
            self.featx = nn.Linear(2592,128)
        else :
        '''
        self.featx = nn.Linear(self.state_dim,400)
        self.featx.weight.data = init_weights(self.featx.weight.data.size())
        
        # Actor network :
        self.actor1 = nn.Linear(400,300)
        self.actor1.weight.data.uniform_(-EPS,EPS)
        self.actor2 = nn.Linear(300,self.action_dim)
        self.actor2.weight.data.uniform_(-EPS,EPS)

        self.use_cuda = use_cuda
        if self.use_cuda :
            self = self.cuda()


    def features(self,x) :
        '''
        if self.CNN['use_cnn'] :
            x1 = F.relu( self.bn1(self.conv1(x) ) )
            x2 = F.relu( self.bn2(self.conv2(x1) ) )
            x3 = F.relu( self.bn3(self.conv3(x2) ) )
            x4 = x3.view( x3.size(0), -1)
            #print(x4.size())
            fx = F.relu( self.featx( x4) )
            # batch x 128 
        else :
        '''
        fx = self.actfn( self.featx( x) )
        # batch x 400
        return fx

    def forward(self, x) :
        fx = self.features(x)
        # batch x 400
        out = self.actfn( self.actor1( fx ) )
        # batch x 300
        out = self.actor2( fx )
        # batch x self.action_dim
        
        #scale the actions :
        unscaled = F.tanh(xx)
        scaled = unscaled * self.action_scaler
        return scaled

    def clone(self):
        cloned = ActorNN(state_dim=self.state_dim,action_dim=self.action_dim,action_scaler=self.action_scaler,CNN=self.CNN,HER=self.HER,actfn=self.actfn, use_cuda=self.use_cuda)
        cloned.load_state_dict( self.state_dict() )
        return cloned

class CriticNN(nn.Module) :
    def __init__(self,state_dim=3,action_dim=2,useCNN={'use_cnn':False,'input_size':3},HER=False,actfn=LeakyReLU,, use_cuda=False  ) :
        super(CriticNN,self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.CNN = useCNN
        # dictionnary with :
        # - 'input_size' : int
        # - 'use_cnn' : bool
        if self.CNN['use_cnn'] :
            self.state_dim = self.CNN['input_size']

        self.HER = HER
        if self.HER :
            self.state_dim *= 2

        self.actfn = actfn
        
        #Features :
        '''
        TODO :
        implement the cloning scheme for this:
        if self.CNN['use_cnn'] :
            self.conv1 = nn.Conv2d(self.state_dim,16, kernel_size=5, stride=2)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = nn.Conv2d(16,32, kernel_size=5, stride=2)
            self.bn2 = nn.BatchNorm2d(32)
            self.conv3 = nn.Conv2d(32,32, kernel_size=5, stride=2)
            self.bn3 = nn.BatchNorm2d(32)
            #self.featx = nn.Linear(448,self.nbr_actions)
            self.featx = nn.Linear(192,128)
        else :
        '''
        self.featx = nn.Linear(self.state_dim,400)
        self.featx.weight.data = init_weights(self.featx.weight.data.size())

        # Critic network :
        ## state value path :
        self.critic1 = nn.Linear(400+self.action_dim,300)
        self.critic1.weight.data = init_weights(self.critic1.weight.data.size())
        
        self.critic2 = nn.Linear(300,1)
        self.critic2.weight.data.uniform_(-EPS*1e-1,EPS*1e-1) 

        self.use_cuda = use_cuda
        if self.use_cuda :
            self = self.cuda()

    def features(self,x) :
        '''
        if self.CNN['use_cnn'] :
            x1 = F.relu( self.bn1(self.conv1(x) ) )
            x2 = F.relu( self.bn2(self.conv2(x1) ) )
            x3 = F.relu( self.bn3(self.conv3(x2) ) )
            x4 = x3.view( x3.size(0), -1)
            
            fx = F.relu( self.featx( x4) )
            # batch x 300 
        else :
        '''
        fx = self.actfn( self.featx(x) )
        # batch x 400
    
        return fx

    def forward(self, x,a) :
        fx = self.features(x)
        # batch x 400
        concat = torch.cat([ fx, a], dim=1)
        # batch x 400+self.action_dim
        out = self.actfn( self.critic1( concat ) )
        # batch x 300
        out = self.critic2(out)
        # batch x 1 
        return out

    def clone(self):
        cloned = CriticNN(state_dim=self.state_dim,action_dim=self.action_dim,CNN=self.CNN,HER=self.HER,actfn=self.actfn, use_cuda=self.use_cuda)
        cloned.load_state_dict( self.state_dict() )
        return cloned



class DeepDeterministicPolicyGradientAlgorithm() :
    def optimize(self,optimizer_critic,optimizer_actor) :
        
        '''
        critic_grad = 0.0
        for p in self.critic.parameters() :
            critic_grad += np.mean(p.grad.cpu().data.numpy())
        print( 'Mean Critic Grad : {}'.format(critic_grad) )
        '''
        
        actor_grad = 0.0
        for p in self.actor.parameters() :
            actor_grad += np.max( np.abs(p.grad.cpu().data.numpy() ) )
        #print( 'Mean Actor Grad : {}'.format(actor_grad) )
        

        #UPDATE THE PR :
        if isinstance(self.memory, PrioritizedReplayBuffer) :
            self.mutex.acquire()
            loss = torch.abs(actor_loss) + torch.abs(critic_loss)
            #loss = torch.abs(actor_loss) #+ torch.abs(critic_loss)
            loss_np = loss.cpu().data.numpy()
            for (idx, new_error) in zip(batch.idx,loss_np) :
                new_priority = self.memory.priority(new_error)
                #print( 'prior = {} / {}'.format(new_priority,self.rBuffer.total()) )
                self.memory.update(idx,new_priority)
            self.mutex.release()

        closs = critic_loss.cpu()
        aloss = actor_loss.cpu()
        
        return closs.data.numpy(), aloss.data.numpy()


class DeepDeterministicPolicyGradientAlgorithm :
    def __init__(self,kwargs) :
        """
        :param kwargs:
            "model_actor": actor model of the agent to use/optimize in this algorithm.
            "model_critic": critic model of the agent to use/optimize in this algorithm.

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

            "nbrTrainIteration": int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]

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

        self.kwargs = kwargs
        self.use_cuda = kwargs["use_cuda"]

        self.model_actor = kwargs["model_actor"]
        self.model_critic = kwargs["model_critic"]

        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)

        if self.use_cuda :
            self.target_actor = self.target_actor.cuda()
            self.target_critic = self.target_critic.cuda()
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
    
        
        self.nbr_worker = kwargs["nbr_worker"]

        if self.kwargs['replayBuffer'] is None :
            if kwargs["use_PER"] :
                self.replayBuffer = PrioritizedReplayBuffer(capacity=kwargs["replay_capacity"],alpha=kwargs["PER_alpha"])
            else :
                self.replayBuffer = ReplayMemory(capacity=kwargs["replay_capacity"])
            #self.kwargs['replayBuffer'] = self.replayBuffer
        else :
            self.replayBuffer = self.kwargs['replayBuffer']

        self.min_capacity = kwargs["min_capacity"]
        self.batch_size = kwargs["batch_size"]

        self.lr = kwargs["lr"]
        self.TAU = kwargs["tau"]
        self.GAMMA = kwargs["gamma"]
        
        self.optimizer_actor = optim.Adam(self.model_actor.parameters(), lr=self.lr*1e-1 )
        self.optimizer_critic = optim.Adam(self.model_critic.parameters(), lr=self.lr )

        self.preprocess = kwargs["preprocess"]

        self.noise = OrnsteinUhlenbeckNoise(self.model_actor.action_dim)
    
    def clone(self) :
        cloned_kwargs = self.kwargs
        cloned_model_actor = self.model_actor.clone()
        cloned_model_critic = self.model_critic.clone()
        self.kwargs['model_actor'] = cloned_model_actor
        self.kwargs['model_actor'] = cloned_model_critic
        cloned = DeepDeterministicPolicyGradientAlgorithm(kwargs=cloned_kwargs)
        return cloned

    def evaluate(self, state,action,target=False) :
        if self.use_cuda :
            state = state.cuda()
            action = action.cuda()
        if ~target :
            qsa = self.critic( state, action).detach()
        else :
            qsa = self.target_critic( state, action).detach()
        return qsa.cpu().data.numpy()

    def update_targets(self):
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)
        
    def optimize_model(self) :
        if len(self.replayBuffer) < self.min_capacity :
            return None
        
        if self.kwargs['use_PER'] :
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

            # Importance Sampling Weighting:
            beta = 1.0
            priorities = Variable( torch.from_numpy( np.array(priorities) ), requires_grad=False).float()
            importanceSamplingWeights = torch.pow( len(self.replayBuffer) * priorities , -beta)
        else :
            # Create Batch with replayMemory :
            transitions = replayBuffer.sample(self.batch_size)
            batch = Transition(*zip(*transitions) )

        next_state_batch = Variable(torch.cat( batch.next_state), requires_grad=False)
        state_batch = Variable( torch.cat( batch.state) , requires_grad=False)
        action_batch = Variable( torch.cat( batch.action) , requires_grad=False)
        reward_batch = Variable( torch.cat( batch.reward ), requires_grad=False ).view((-1,1))
        done_batch = [ 0.0 if batch.done[i] else 1.0 for i in range(len(batch.done)) ]
        done_batch = Variable( torch.FloatTensor(done_batch), requires_grad=False ).view((-1,1))

        if self.use_cuda :
            if self.kwargs['use_PER']: importanceSamplingWeights = importanceSamplingWeights.cuda()
            next_state_batch = next_state_batch.cuda()
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            done_batch = done_batch.cuda()


        self.optimizer_critic.zero_grad()

        # Critic :
        # sample action from next_state, without gradient repercusion :
        next_taction = self.target_actor(next_state_batch).detach()
        # evaluate the next state action over the target, without repercusion :
        next_tqsa = torch.squeeze( self.target_critic( next_state_batch, next_taction).detach() ).view((-1))
        # Critic loss :
        ## y_true :
        y_true = reward_batch + (1.0-done_batch)*self.gamma*next_tqsa
        ## y_pred :
        y_pred = torch.squeeze( self.critic(state_batch,action_batch) )
        
        # Compute loss:
        diff = y_true - y_pred
        if self.kwargs['use_PER'] :
            diff_squared = importanceSamplingWeights.unsqueeze(1) * diff.pow(2.0)
        else :
            diff_squared = diff.pow(2.0)
        critic_loss_per_item = diff_squared
        critic_loss = torch.mean( diff_squared)
        critic_loss.backward()
        '''
        #critic_loss = F.smooth_l1_loss(y_pred,y_true)
        criterion = nn.MSELoss()
        critic_loss = criterion(y_pred,y_true)
        critic_loss.backward()
        '''
        #weight decay :
        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model_critic.parameters()])
        weights_decay_loss.backward()

        #clamping :
        #torch.nn.utils.clip_grad_norm(self.model_critic.parameters(),50)             
        self.optimizer_critic.step()
        

        ###################################
        
        # Actor :
        #before optimization :
        self.optimizer_actor.zero_grad()
        
        '''
        #predict action :
        pred_action = self.model_actor(state_batch) 
        var_action = Variable( pred_action.cpu().data, requires_grad=True)
        if self.use_cuda :
            var_action = var_action.cuda()
        pred_qsa = self.model_critic(state_batch, var_action_c)
        #predict associated qvalues :
        gradout = torch.ones(pred_qsa.size())
        if self.use_cuda:
            gradout = gradout.cuda()
        pred_qsa.backward( gradout )

        gradcritic = var_action.grad.data
        pred_action.backward( -gradcritic)
        '''
        actor_loss_per_item = -self.model_critic(state_batch, self.model_actor(state_batch) )
        actor_loss = actor_loss_per_item.mean()
        actor_loss.backward()

        #weight decay :
        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model_actor.parameters()])
        weights_decay_loss.backward()

        #clamping :
        #torch.nn.utils.clip_grad_norm(self.model_actor.parameters(),50)             
        self.optimizer_actor.step()

        
        ###################################

        if self.kwargs['use_PER']:
            #UPDATE THE PER :
            loss = torch.abs(actor_loss_per_item) + torch.abs(critic_loss_per_item)
            loss_np = loss.cpu().data.numpy()
            for (idx, new_error) in zip(batch.idx,loss_np) :
                new_priority = self.replayBuffer.priority(new_error)
                self.replayBuffer.update(idx,new_priority)

        return loss_np

    def handle_experience(self, experience):
        '''
        This function is only called during training.
        It stores experience in the replay buffer.

        :param experience: EXP object containing the current, relevant experience.
        '''
        if self.kwargs["use_PER"]:
            init_sampling_priority = self.replayBuffer.priority(torch.abs(experience.reward).cpu().numpy() )
            self.replayBuffer.add(experience, init_sampling_priority)
        else:
            self.replayBuffer.push(experience)

    def train(self, iteration=1):
        for t in range(iteration):
            lossnp = self.optimize_model()
            self.update_targets()





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
        return self.act(x=self.preprocess_function(state), exploitation=not(self.training), exploration_noise=None)
        
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
        """
        TODO : decide whether to launch the training automatically or do it manually.
        So far it is being done manually...
        """
        from .agent_hook import AgentHook
        cloned = AgentHook(self, training=training, path=path)
        return cloned

def build_DDPG_Agent(state_space_size=32,
                        action_space_size=3,
                        learning_rate=1e-3,
                        num_worker=1,
                        nbrTrainIteration=1,
                        action_scaler=1.0,
                        use_PER=False,
                        alphaPER=0.6,
                        MIN_MEMORY=5e1,
                        use_cuda=False):
    kwargs = dict()
    """
    :param kwargs:
        "model_actor": actor model of the agent to use/optimize in this algorithm.
        "model_critic": critic model of the agent to use/optimize in this algorithm.
        
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

        "nbrTrainIteration": int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]

        "nbr_worker": int to specify whether to use the Distributed variant of DQN and how many worker to use [default: nbr_worker=1].

        "nbr_actions":
        "actfn":
        "useCNN":.
        
    """

    """
    TODO : implement CNN usage for DQN...
    """
    useCNN = {'use_cnn':use_cnn, 'input_size':state_space_size}
    if useCNN['use']:
        preprocess = T.Compose([T.ToPILImage(),
                    T.Scale(64, interpolation=Image.CUBIC),
                    T.ToTensor() ] )
    else :
        preprocess = T.Compose([
                    T.ToTensor() ] )

    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 1e-3
    
    #HER :
    k = 2
    strategy = 'future'
    singlegoal = False
    HER = {'k':k, 'strategy':strategy,'use_her':use_HER,'singlegoal':singlegoal}
    
    kwargs['nbrTrainIteration'] = nbrTrainIteration
    kwargs["action_dim"] = action_space_size
    kwargs["state_dim"] = state_space_size
    kwargs["action_scaler"] = action_scaler
    
    kwargs["actfn"] = LeakyReLU
    kwargs["useCNN"] = useCNN
    
    # Create model architecture:
    actor = ActorNN(state_dim=state_space_size,action_dim=action_space_size,action_scaler=action_scaler,CNN=CNN,HER=HER['use_her'])
    actor.share_memory()
    critic = CriticNN(state_dim=state_space_size,action_dim=action_space_size,dueling=dueling,CNN=CNN,HER=HER['use_her'])
    critic.share_memory()
    print("DDPG model initialized: OK")
    kwargs["model_actor"] = actor
    kwargs["model_critic"] = critic
    
    name = "DDPG"
    if dueling : name = 'Dueling'+name
    model_path = './'+name
    #model_path += '-MSELoss-w'+str(num_worker)+'-lr'+str(lr)+'-b'+str(BATCH_SIZE)+'-tau'+str(TAU)+'-m'+str(memoryCapacity)+'/'
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

    kwargs["lr"] = learning_rate
    kwargs["tau"] = TAU
    kwargs["gamma"] = GAMMA

    kwargs["preprocess"] = preprocess
    kwargs["nbr_worker"] = num_worker

    kwargs['replayBuffer'] = None

    DeepDeterministicPolicyGradient_algo = DeepDeterministicPolicyGradientAlgorithm(kwargs=kwargs)

    return DDPGAgent(algorithm=DeepDeterministicPolicyGradient_algo)

if __name__ == "__main__":
    build_DDPG_Agent(use_PER=False)
