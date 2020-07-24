from copy import deepcopy
import itertools
from torch.optim import Adam
import torch
import torch.nn as nn

from regym.networks.utils import soft_update
from regym.rl_algorithms.replay_buffers import ReplayBuffer, EXP 
from regym.rl_algorithms.SAC.soft_actor_critic_losses import compute_q_critic_loss, compute_pi_actor_loss


class SoftActorCriticAlgorithm():

    def __init__(self, use_cuda: bool,
                 learning_rate: float,
                 tau: float,
                 alpha: float,
                 gamma: float,
                 batch_size: int,
                 memory_size: int,  # TODO: think, Maybe we can retrieve this from memory buffer?
                 pi_actor: nn.Module,
                 q_critic_1: nn.Module,
                 q_critic_2: nn.Module,
                 replay_buffer: ReplayBuffer):
        '''
        TODO: Document

        :param use_cuda: bool,
        :param learning_rate: float,
        :param tau: float,
        :param alpha: float,
        :param gamma: float,
        :param batch_size: int,
        :param memory_size: int,
        :param replay_buffer: ReplayBuffer
        '''
        self.use_cuda = use_cuda  # TODO: implement this!
        self.learning_rate = learning_rate
        self.tau = tau
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.replay_buffer = replay_buffer

        # TODO: figure out model for pi and (Q1, Q2) and their respective target
        self.pi_actor = pi_actor
        self.q_critic_1 = q_critic_1
        self.q_critic_2 = q_critic_2

        self.q_critic_1_targ = deepcopy(q_critic_1)
        self.q_critic_2_targ = deepcopy(q_critic_2)

        # Useful when we compute the minium Q values in q_loss
        # TODO: why is q_params empty?
        self.q_params = lambda: itertools.chain(
                self.q_critic_1.parameters(), self.q_critic_2.parameters())

        self.pi_actor_optimizer = Adam(self.pi_actor.parameters(), lr=learning_rate)
        self.q_critic_optimizer = Adam(self.q_params(), lr=learning_rate)


        # Number of updates to policy and Q functions
        self.iteration_count = 0

    def update(self):
        transitions, batch = self.sample_from_replay_buffer(self.batch_size)

        next_state_batch, state_batch, action_batch, reward_batch, \
        not_done_batch = self.create_tensors_for_optimization(
                batch, use_cuda=self.use_cuda)

        self.compute_and_propagate_losses(state_batch, action_batch, reward_batch,
                                          next_state_batch, not_done_batch)
        self.iteration_count += 1


    def compute_and_propagate_losses(self, state_batch: torch.Tensor,
                                     action_batch: torch.Tensor,
                                     reward_batch: torch.Tensor,
                                     next_state_batch: torch.Tensor,
                                     not_done_batch: torch.Tensor):
        # First run one gradient descent step for Q1 and Q2
        loss_q = compute_q_critic_loss(
                state_batch, action_batch, reward_batch,
                next_state_batch, not_done_batch,
                self.q_critic_1, self.q_critic_2,
                self.q_critic_1_targ, self.q_critic_2_targ,
                self.pi_actor, self.alpha, self.gamma,
                self.iteration_count)

        from regym.util.nn_debugging import plot_gradient_flow, plot_backwards_graph
        self.q_critic_optimizer.zero_grad()
        loss_q.backward()
        self.q_critic_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params(): p.requires_grad = False

        # Next run one gradient descent step for pi.
        loss_pi = compute_pi_actor_loss(state_batch,
                self.q_critic_1, self.q_critic_2, self.pi_actor, self.alpha,
                self.iteration_count)

        self.pi_actor_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_actor_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params(): p.requires_grad = True

        # Finally, *softly* update target networks
        soft_update(fromm=self.q_critic_1, to=self.q_critic_1_targ, tau=self.tau)
        soft_update(fromm=self.q_critic_2, to=self.q_critic_2_targ, tau=self.tau)

    def sample_from_replay_buffer(self, batch_size: int):
        transitions = self.replay_buffer.sample(self.batch_size)
        # Rewards look weird (960 instead of 32), and not_done_batch should be (32), not [1, 32]
        batch = EXP(*zip(*transitions))
        return transitions, batch

    def create_tensors_for_optimization(self, batch, use_cuda: bool):
        '''
        TODO: document
        '''
        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward).view((-1, 1))
        not_done_batch = [float(not batch.done[i]) for i in range(len(batch.done))]
        not_done_batch = torch.FloatTensor(not_done_batch).view((-1, 1))

        if use_cuda:
            next_state_batch = next_state_batch.cuda()
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            not_done_batch = not_done_batch.cuda()
        return next_state_batch, state_batch, action_batch, \
               reward_batch, not_done_batch
