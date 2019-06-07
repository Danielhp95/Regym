import torch
import torch.nn as nn
import torch.nn.functional as F
from ..networks.ppo_network_utils import layer_init

from functools import reduce


class ReinforceAlgorithm():

    def __init__(self, policy_model_input_dim, policy_model_output_dim, learning_rate):
        self.learning_rate = learning_rate
        self.model = FullyConnectedFeedForward(policy_model_input_dim, policy_model_output_dim, hidden_units=(64,))

    def train(self):
        # Compute gradient
        # Update parameters
        pass


class FullyConnectedFeedForward(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_units=(64,), gate=F.relu):
        super(FullyConnectedFeedForward, self).__init__()
        dimensions = (input_dim,) + hidden_units + (output_dim,)
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out))
                                     for dim_in, dim_out in zip(dimensions[:-1], dimensions[1:])])
        self.gate = gate

    def forward(self, x):
        x = torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)
        logits = reduce(lambda acc, layer: self.gate(layer(acc)), self.layers, x)
        distribution = torch.distributions.Categorical(logits=logits)
        action = distribution.sample(sample_shape=(logits.size(0),))
        log_probability = distribution.log_prob(action)
        return {'action': action, 'log_probability': log_probability}
