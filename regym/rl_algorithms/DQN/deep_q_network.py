from typing import Dict
import copy

import torch
import torch.optim as optim
from torch.autograd import Variable

from regym.rl_algorithms.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer, EXP, EXPPER
from regym.networks.utils import hard_update
from regym.networks.utils import compute_weights_decay_loss
from regym.rl_algorithms.DQN.dqn_loss import compute_loss


class DeepQNetworkAlgorithm():
    def __init__(self, kwargs: Dict[str, object],
                 model: torch.nn.Module, target_model: torch.nn.Module = None):
        """
        :param kwargs:
            "use_cuda": boolean to specify whether to use CUDA.
            "replay_capacity": int, capacity of the replay buffer to use.
            "min_capacity": int, minimal capacity before starting to learn.
            "batch_size": int, batch size to use [default: batch_size=256].
            "use_PER": boolean to specify whether to use a Prioritized Experience Replay buffer.
            "PER_alpha": float, alpha value for the Prioritized Experience Replay buffer.
            "lr": float, learning rate.
            "tau": float, target update rate.
            "gamma": float, Q-learning gamma rate.
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
        :param model: model of the agent to use/optimize in this algorithm.

        """

        self.kwargs = kwargs
        self.use_cuda = kwargs["use_cuda"]

        # DQN extensions
        self.use_double = kwargs['double']
        self.use_dueling = kwargs['dueling']

        # Setting neural network models (target and normal model)
        self.model = model
        self.target_model = target_model if target_model else copy.deepcopy(self.model)

        if self.use_cuda:
            self.model = self.model.cuda()
            self.target_model = self.target_model.cuda()

        self.target_model.share_memory()

        # Replay buffer parameters
        if kwargs["use_PER"]:
            self.replayBuffer = PrioritizedReplayBuffer(capacity=kwargs["replay_capacity"], alpha=kwargs["PER_alpha"])
        else:
            self.replayBuffer = ReplayBuffer(capacity=kwargs["replay_capacity"])

        self.min_capacity = kwargs["min_capacity"]
        self.batch_size = kwargs["batch_size"]

        # Target network update parameters
        self.TAU = kwargs["tau"]
        self.target_update_interval = int(1.0/self.TAU)
        self.target_update_count = 0

        # Learning rate parameters
        self.lr = kwargs["lr"]
        self.GAMMA = kwargs["gamma"]
        self.optimizer: torch.optim = optim.Adam(self.model.parameters(), lr=self.lr)

        # PreprocessFunction
        self.preprocess = kwargs["preprocess"]

        # Exploration rate parameters
        self.epsend = kwargs['epsend']
        self.epsstart = kwargs['epsstart']
        self.epsdecay = kwargs['epsdecay']

    def clone(self):
        cloned_kwargs = self.kwargs
        cloned_model = self.model.clone()
        cloned_model.share_memory()
        cloned_target_model = self.target_model.clone()
        cloned_target_model.share_memory()
        cloned = DeepQNetworkAlgorithm(kwargs=cloned_kwargs, model=cloned_model, target_model=cloned_target_model)
        return cloned

    def is_ready_to_train(self):
        return self.replayBuffer.current_size >= self.min_capacity

    def optimize_model(self, gradient_clamping_value=None):
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
        # TODO: worry about this later
        # if self.kwargs['use_PER']:
        #     # Create batch with PrioritizedReplayBuffer/PER:
        #     transitions, importance_sampling_weights = self.replayBuffer.sample(self.batch_size)
        #     batch = EXPPER(*zip(*transitions))
        #     importance_sampling_weights = torch.from_numpy(importance_sampling_weights)
        #     if self.use_cuda:
        #         importance_sampling_weights = importance_sampling_weights.cuda()

        self.optimizer.zero_grad()
        transitions, batch = self.sample_from_replay_buffer(self.batch_size)

        next_state_batch, state_batch, action_batch, reward_batch, \
        non_terminal_batch = self.create_tensors_for_optimization(batch,
                                                                  use_cuda=self.use_cuda)

        dqn_loss = compute_loss(states=state_batch,
                                actions=action_batch,
                                next_states=next_state_batch,
                                rewards=reward_batch,
                                non_terminals=non_terminal_batch,
                                model=self.model,
                                target_model=self.target_model,
                                gamma=self.GAMMA,
                                use_double=self.use_double,
                                use_dueling=self.use_dueling,
                                iteration_count=self.target_update_count)

        dqn_loss.backward()

        if gradient_clamping_value is not None:
            torch.nn.utils.clip_grad_norm(self.model.parameters(), gradient_clamping_value)

        self.optimizer.step()

        # TODO: Worry about this later
        #loss_per_item = dqn_loss
        #loss_np = loss_per_item.cpu().data.numpy()
        #if self.kwargs['use_PER']:
        #    for (idx, new_error) in zip(batch.idx, loss_np):
        #        new_priority = self.replayBuffer.priority(new_error)
        #        self.replayBuffer.update(idx, new_priority)

        #return loss_np

    def create_tensors_for_optimization(self, batch, use_cuda: bool):
        '''
        TODO: document
        '''
        next_state_batch = Variable(torch.cat(batch.next_state), requires_grad=False)
        state_batch = Variable(torch.cat(batch.state), requires_grad=False)
        action_batch = Variable(torch.cat(batch.action), requires_grad=False)
        reward_batch = Variable(torch.cat(batch.reward), requires_grad=False).view((-1, 1))
        non_terminal_batch = [float(not batch.done[i]) for i in range(len(batch.done))]
        non_terminal_batch = Variable(torch.FloatTensor(non_terminal_batch), requires_grad=False).view((-1, 1))

        if use_cuda:
            next_state_batch = next_state_batch.cuda()
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            non_terminal_batch = non_terminal_batch.cuda()

        return next_state_batch, state_batch, action_batch, \
               reward_batch, non_terminal_batch

    def sample_from_replay_buffer(self, batch_size: int):
        transitions = self.replayBuffer.sample(self.batch_size)
        batch = EXP(*zip(*transitions))
        return transitions, batch

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

    def train(self, iterations: int):
        self.target_update_count += iterations
        for t in range(iterations):
            _ = self.optimize_model()

        def weight_decay_closure():
            self.optimizer.zero_grad()
            weights_decay_loss = compute_weights_decay_loss(self.model)
            weights_decay_loss.backward()
            return weights_decay_loss
        self.optimizer.step(weight_decay_closure)

        # Update target network
        if self.target_update_count % self.target_update_interval == 0:
            hard_update(self.target_model, self.model)
