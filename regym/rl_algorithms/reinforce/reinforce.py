import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from regym.rl_algorithms.networks.utils import layer_init

from functools import reduce


class ReinforceAlgorithm():

    def __init__(self, policy_model_input_dim, policy_model_output_dim, learning_rate, adam_eps):
        self.learning_rate = learning_rate
        self.model = FullyConnectedFeedForward(policy_model_input_dim, policy_model_output_dim, hidden_units=(16,))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=adam_eps)

    def train(self, trajectories):
        def closure():
            self.optimizer.zero_grad()
            loss = -1. * self.compute_policy_utility_gradient(trajectories) # check that a gradient function exists
            loss.backward()
            return loss
        self.optimizer.step(closure)

    def compute_policy_utility_gradient(self, trajectories):
        gradient = reduce(lambda acc, trajectory: acc + self.compute_trajectory_loss(trajectory), trajectories, 0)
        normalized_gradient = gradient / len(trajectories) # make sure that tensors have type.float
        return normalized_gradient

    def compute_trajectory_loss(self, trajectory):
        # Normalize reward? normalized_episode_reward = (episode_reward - episode_reward.mean()) / std
        episode_reward = self.cummulative_trajectory_reward(trajectory)
        return sum([log_a * episode_reward for (s, a, log_a, r, succ_s) in trajectory])

    def cummulative_trajectory_reward(self, trajectory):
        return sum([reward for (s, a, log_a, reward, succ_s) in trajectory])


class FullyConnectedFeedForward(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_units=(32,), gate=F.relu):
        super(FullyConnectedFeedForward, self).__init__()
        dimensions = (input_dim,) + hidden_units + (output_dim,)
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out))
                                     for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])])
        self.gate = gate

    def forward(self, x):
        x = torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor).cpu()
        last_layer_output = reduce(lambda acc, layer: self.gate(layer(acc)), self.layers, x)
        action_probabilities = F.softmax(last_layer_output)
        distribution = torch.distributions.Categorical(probs=action_probabilities)
        action = distribution.sample(sample_shape=(action_probabilities.size(0),))
        log_probability = distribution.log_prob(action)
        return {'action': action, 'action_log_probability': log_probability}
