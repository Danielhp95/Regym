import numpy as np
import copy

import torch
import torch.optim as optim
from torch.autograd import Variable

from ..replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, EXP, EXPPER
from ..networks import  soft_update, hard_update, LeakyReLU, ActorNN, CriticNN

class OrnsteinUhlenbeckNoise :
    def __init__(self, dim,mu=0.0, theta=0.15, sigma=0.2) :
        self.dim = dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.X = np.ones(self.dim)*self.mu

    def setSigma(self,sigma):
        self.sigma = sigma
    
    def sample(self) :
        dx = self.theta * ( self.mu - self.X)
        dx += self.sigma *  np.random.randn( self.dim )
        self.X += dx
        return self.X


class DeepDeterministicPolicyGradientAlgorithm :
    def __init__(self,kwargs, models) :
        """
        :param kwargs:
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
        :param models: dict
            "actor": actor model of the agent to use/optimize in this algorithm.
            "critic": critic model of the agent to use/optimize in this algorithm.
            
        """

        self.kwargs = kwargs
        self.use_cuda = kwargs["use_cuda"]

        self.model_actor = models["actor"]
        self.model_critic = models["critic"]

        self.target_actor = copy.deepcopy(self.model_actor)
        self.target_critic = copy.deepcopy(self.model_critic)

        if self.use_cuda :
            self.target_actor = self.target_actor.cuda()
            self.target_critic = self.target_critic.cuda()
        hard_update(self.target_actor, self.model_actor)
        hard_update(self.target_critic, self.model_critic)

        if self.kwargs['replayBuffer'] is None :
            if kwargs["use_PER"] :
                self.replayBuffer = PrioritizedReplayBuffer(capacity=kwargs["replay_capacity"],alpha=kwargs["PER_alpha"])
            else :
                self.replayBuffer = ReplayBuffer(capacity=kwargs["replay_capacity"])
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
        cloned = DeepDeterministicPolicyGradientAlgorithm(kwargs=cloned_kwargs, models={"actor":cloned_model_actor, "critic":cloned_model_critic})
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
        
    def optimize_model(self, gradient_clamping_value=None) :
        """
        
        Optional: at the end, update the Prioritized Experience Replay buffer with new priorities.
        
        :param gradient_clamping_value: if None, the gradient is not clamped, 
                                        otherwise a positive float value is expected as a clamping value 
                                        and gradients are clamped.
        :returns loss_np: numpy scalar of the estimated loss function.
        """

        if len(self.replayBuffer) < self.min_capacity :
            return None
        
        if self.kwargs['use_PER'] :
            #Create batch with PrioritizedReplayBuffer/PER:
            prioritysum = self.replayBuffer.total()
            low = 0.0
            step = (prioritysum-low) / self.batch_size
            try:
                randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(self.batch_size))
            except Exception as e :
                print( prioritysum, step)
                raise e
            
            batch = list()
            priorities = []
            for i in range(self.batch_size):
                try :
                    el = self.replayBuffer.get(randexp[i])
                    priorities.append( el[1] )
                    batch.append(el)
                except TypeError as e :
                    continue
            
            batch = EXPPER( *zip(*batch) )

            # Importance Sampling Weighting:
            beta = 1.0
            priorities = Variable( torch.from_numpy( np.array(priorities) ), requires_grad=False).float()
            importanceSamplingWeights = torch.pow( len(self.replayBuffer) * priorities , -beta)
        else :
            # Create Batch with replayBuffer :
            transitions = replayBuffer.sample(self.batch_size)
            batch = EXP(*zip(*transitions) )

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
        #weight decay :
        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model_critic.parameters()])
        weights_decay_loss.backward()

        if gradient_clamping_value is not None :
            torch.nn.utils.clip_grad_norm(self.model.parameters(),gradient_clamping_value)

        self.optimizer_critic.step()
        

        ###################################
        
        # Actor :
        #before optimization :
        self.optimizer_actor.zero_grad()
        
        actor_loss_per_item = -self.model_critic(state_batch, self.model_actor(state_batch) )
        actor_loss = actor_loss_per_item.mean()
        actor_loss.backward()

        #weight decay :
        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model_actor.parameters()])
        weights_decay_loss.backward()

        if gradient_clamping_value is not None :
            torch.nn.utils.clip_grad_norm(self.model.parameters(),gradient_clamping_value)

        self.optimizer_actor.step()

        
        ###################################

        loss = torch.abs(actor_loss_per_item) + torch.abs(critic_loss_per_item)
        loss_np = loss.cpu().data.numpy()
        if self.kwargs['use_PER']:
            for (idx, new_error) in zip(batch.idx,loss_np) :
                new_priority = self.replayBuffer.priority(new_error)
                self.replayBuffer.update(idx,new_priority)

        return loss_np

    def handle_experience(self, experience):
        '''
        This function is only called during training.
        It stores experience in the replay buffer.

        :param experience: EXP/EXPPER object containing the current, relevant experience.
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

