import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from ..replay_buffers import EXP, EXPPER
from .deep_q_network import DeepQNetworkAlgorithm


class DoubleDeepQNetworkAlgorithm(DeepQNetworkAlgorithm) :

    def clone(self) :
        cloned_kwargs = self.kwargs
        cloned_model = self.model.clone()
        self.kwargs['model'] = cloned_model
        cloned = DoubleDeepQNetworkAlgorithm(kwargs=cloned_kwargs)
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
           # Create Batch with replayMemory :
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
        if self.kwargs['use_PER']:
            diff_squared = importanceSamplingWeights * diff.pow(2.0)
        else :
            diff_squared =diff.pow(2.0)
        loss_per_item = diff_squared
        loss = torch.mean( diff_squared)
        loss.backward()

        if gradient_clamping_value is not None :
            torch.nn.utils.clip_grad_norm(self.model.parameters(),gradient_clamping_value)

        weights_decay_lambda = 1e-0
        weights_decay_loss = weights_decay_lambda * 0.5*sum( [torch.mean(param*param) for param in self.model.parameters()])
        weights_decay_loss.backward()

        self.optimizer.step()

        if self.kwargs['use_PER']:
            loss_np = loss_per_item.cpu().data.numpy()
            for (idx, new_error) in zip(batch.idx,loss_np) :
                new_priority = self.replayBuffer.priority(new_error)
                self.replayBuffer.update(idx,new_priority)

        return loss_np
