import numpy as np
import copy

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from ..replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, EXP, EXPPER
from ..networks import  hard_update, LeakyReLU, DQN, DuelingDQN 


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
            "tau": float, target update rate [default: tau=1e-3].
            "gamma": float, Q-learning gamma rate [default: gamma=0.999].
            "preprocess": preprocessing function/transformation to apply to observations [default: preprocess=T.ToTensor()]
            "nbrTrainIteration": int, number of iteration to train the model at each new experience. [default: nbrTrainIteration=1]
            "epsstart": starting value of the epsilong for the epsilon-greedy policy.
            "epsend": asymptotic value of the epsilon for the epsilon-greedy policy.
            "epsdecay": rate at which the epsilon of the epsilon-greedy policy decays.

            "dueling": boolean specifying whether to use Dueling Deep Q-Network architecture
            "double": boolean specifying whether to use Double Deep Q-Network algorithm.
            "nbr_actions": number of dimensions in the action space.
            "actfn": activation function to use in between each layer of the neural networks.
            "state_dim": number of dimensions in the state space.
        """
        
        self.kwargs = kwargs
        self.use_cuda = kwargs["use_cuda"]

        self.model = kwargs["model"]
        if self.use_cuda:
            self.model = self.model.cuda()

        self.target_model = copy.deepcopy(self.model)
        hard_update(self.target_model,self.model)
        if self.use_cuda :
            self.target_model = self.target_model.cuda()

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
        self.target_update_interval = int(1.0/self.TAU)
        self.target_update_count = 0
        self.GAMMA = kwargs["gamma"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr )

        self.preprocess = kwargs["preprocess"]

        self.epsend = kwargs['epsend']
        self.epsstart = kwargs['epsstart']
        self.epsdecay = kwargs['epsdecay']

    def clone(self) :
        cloned_kwargs = self.kwargs
        cloned_model = self.model.clone()
        self.kwargs['model'] = cloned_model
        cloned = DeepQNetworkAlgorithm(kwargs=cloned_kwargs)

        return cloned

    def optimize_model(self,gradient_clamping_value=None) :
        """
        1) Estimate the gradients of the loss with respect to the
        current learner model on a batch of experiences sampled
        from the replay buffer.
        2) Backward the loss.
        3) Update the weights with the optimizer.
        4) Optional: Update the Prioritized Experience Replay buffer with new priorities.
        
        :param gradient_clamping_value: if None, the gradient is not clamped, 
                                        otherwise a positive float value is expected as a clamping value 
                                        and gradients are clamped.
        :returns loss_np: numpy scalar of the estimated loss function.
        """

        if len(self.replayBuffer) < self.min_capacity :
            return None

        if self.kwargs['use_PER'] :
            #Create batch with PrioritizedReplayBuffer/PER:
            transitions, importanceSamplingWeights = self.replayBuffer.sample(self.batch_size)
            batch = EXPPER( *zip(*transitions) )
            importanceSamplingWeights = torch.from_numpy(importanceSamplingWeights)
        else :
            # Create Batch with replayBuffer :
            transitions = self.replayBuffer.sample(self.batch_size)
            batch = EXP( *zip(*transitions) )
            
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
        if self.kwargs['use_PER'] :
            diff_squared = importanceSamplingWeights.unsqueeze(1) * diff.pow(2.0)
        else :
            diff_squared = diff.pow(2.0)
        loss_per_item = diff_squared
        loss = torch.mean( diff_squared)
        loss.backward()

        if gradient_clamping_value is not None :
            torch.nn.utils.clip_grad_norm(self.model.parameters(),gradient_clamping_value)

        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model.parameters()])
        weights_decay_loss.backward()

        self.optimizer.step()

        loss_np = loss_per_item.cpu().data.numpy()
        if self.kwargs['use_PER']:
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
        self.target_update_count += iteration
        for t in range(iteration):
            lossnp = self.optimize_model()
            
        if self.target_update_count > self.target_update_interval:
            self.target_update_count = 0
            hard_update(self.target_model,self.model)

