import torch
from torch import nn
import torch.nn.functional as F


class AutoEncoderEnvironmentModel(nn.Module):
    def __init__(self, encoder, decoder, observation_shape, num_actions, reward_size, use_cuda=False):
        """
        :param encoder:
        :param decoder:
        :param observation_shape: shape depth x height x width.
        :param num_actions: number of actions that are available in the environment.
        :param reward_size: number of dimensions of the reward vector. 
                            Eventhough OpenAI Gym Interface always provides
                            scalar reward function, it might be interesting 
                            in some other environments to predict a vector of
                            reward functions.
        :param conv_dim: number of convolution kernels to use for each conv layer.
        """
        super(AutoEncoderEnvironmentModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.reward_size = reward_size
        self.use_cuda = use_cuda

        self.reward_fc = nn.Linear(self.encoder.get_feature_shape(), self.reward_size)

        if self.use_cuda: self = self.cuda()

    def forward(self, observations, actions):
        """
        Returns the next observation and reward, given :param observations: 
        and :param actions:.
        :param observations: observation tensor of shape batch x (depth x height x width)/hidden_units.
        :param actions: action indexes tensor of shape batch x 1.
        """

        batch_size = observations.size(0)
        
        actions = actions.long()
        if self.use_cuda: actions = actions.cuda()

        onehot_actions = torch.zeros(batch_size, self.num_actions, *(observations.size())[2:])
        if self.use_cuda: onehot_actions = onehot_actions.cuda()

        onehot_actions[range(batch_size), actions] = 1
        inputs = torch.cat([observations, onehot_actions], 1)
        
        latents = self.encoder(inputs)
        
        output = self.decoder(latents)

        reward = self.reward_fc(latents)

        return output, reward 

