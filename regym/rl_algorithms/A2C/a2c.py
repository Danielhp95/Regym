import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from regym.networks.utils import layer_init

from functools import reduce


class A2CAlgorithm():

    def __init__(self, model: nn.Module,
                 discount_factor: float,
                 n_steps: int,
                 learning_rate: float,
                 adam_eps: float):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.n_steps = n_steps
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=adam_eps)

    def train(self, samples, bootstrapped_reward):
        rewards                  = np.array([reward for (s, a, log_a, reward, state_value, succ_s, done) in samples])
        q_values                 = self.compute_temporal_differences_targets(rewards, bootstrapped_reward)
        state_values             = torch.cat([state_value for (s, a, log_a, reward, state_value, succ_s, done) in samples])
        log_action_probabilities = torch.cat([log_a for (s, a, log_a, reward, state_value, succ_s, done) in samples])

        def closure():
            self.optimizer.zero_grad()
            policy_loss = -1. * self.compute_policy_utility_gradient(log_action_probabilities, q_values, state_values)
            value_loss  = nn.MSELoss()(state_values.squeeze(), q_values)
            (policy_loss + value_loss).backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.5)
            return (policy_loss + value_loss)
        self.optimizer.step(closure)

    def compute_policy_utility_gradient(self, log_action_probabilities, q_values, state_values):
        advantages = (q_values - state_values.squeeze()).detach()
        return torch.mean(log_action_probabilities.squeeze() * advantages)

    def compute_temporal_differences_targets(self, rewards, bootstrapped_reward):
        discounted_rewards = np.zeros_like(rewards)
        running_add = bootstrapped_reward
        for t in reversed(range(0, len(rewards))):
            running_add = rewards[t] + self.discount_factor * running_add
            discounted_rewards[t] = running_add
        return torch.from_numpy(discounted_rewards).type(torch.FloatTensor)


class FullyConnectedFeedForward(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_units=(32,), gate=F.relu):
        super(FullyConnectedFeedForward, self).__init__()
        dimensions = (input_dim,) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out))
                                     for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])])
        self.policy_head_layer = layer_init(nn.Linear(hidden_units[-1], output_dim))
        self.value_head_layer = layer_init(nn.Linear(hidden_units[-1], 1))
        self.gate = gate

    def forward(self, x):
        x = torch.Tensor(x).unsqueeze(0).type(torch.FloatTensor).cpu()
        last_layer_output = reduce(lambda acc, layer: self.gate(layer(acc)), self.layers, x)
        # Policy head
        action, log_probability = self.policy_head(self.gate(self.policy_head_layer(last_layer_output)))
        # Value head
        state_value = self.value_head_layer(last_layer_output)
        return {'a': action,
                'log_probs': log_probability,
                'V': state_value}

    def policy_head(self, last_layer_output):
        action_probabilities = F.softmax(last_layer_output)
        distribution = torch.distributions.Categorical(probs=action_probabilities)
        action = distribution.sample(sample_shape=(action_probabilities.size(0),))
        log_probability = distribution.log_prob(action)
        return action, log_probability
