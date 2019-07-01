import torch
from torch import nn
import torch.nn.functional as F


class EnvironmentModel(nn.Module):
    def __init__(self, observation_shape, num_actions, reward_size, conv_dim=32):
        """
        :param observation_shape: shape depth x height x width.
        :param num_actions: number of actions that are available in the environment.
        :param reward_size: size of the reward vector.
        :param conv_dim: number of convolution kernels to use per layer.
        """
        super(EnvironmentModel, self).__init__()

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.reward_size = reward_size
        self.conv_dim = conv_dim

        dim = self.observation_shape[1]

        k=8
        stride=8
        pad=0
        self.conv1 = nn.Conv2d(in_channels=self.observation_shape[0], out_channels=self.conv_dim,
            kernel_size=k, stride=stride, padding=pad)
        dim = (dim-k+2*pad)//stride +1
            
        k=3
        stride=1
        pad=0
        self.conv2 = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim,
            kernel_size=k, stride=stride, padding=pad)
        dim = (dim-k+2*pad)//stride +1
        
        self.conv3 = nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim,
            kernel_size=k, stride=stride, padding=pad)
        dim = (dim-k+2*pad)//stride +1
        
        self.output_conv = nn.TransposeConv2d(in_channels=self.conv_dim, out_channels=self.observation_shape[0],
            kernel_size=3, stride=8, padding=0)
        
        k=3
        stride=1
        pad=0
        self.reward_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim,
                kernel_size=k, stride=stride, padding=pad),
            nn.MaxPool(kernel_size=32,stride=3)
            )
        dim = (dim-k+2*pad)//stride +1
        dim = (dim-32)//3+1
        
        self.reward_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.conv_dim, out_channels=self.conv_dim,
                kernel_size=k, stride=stride, padding=pad),
            nn.MaxPool(kernel_size=32,stride=3)
            )
        dim = (dim-k+2*pad)//stride +1
        dim = (dim-32)//3+1
        
        self.reward_fc = nn.Linear(dim*dim*self.conv_dim, self.reward_size)

    def forward(self, observations, actions):
        """
        Returns the next observation and reward, given :param observations: 
        and :param actions:.
        :param observations: observation tensor of shape batch x depth x height x width.
        :param actions: action indexes tensor of shape batch x 1.
        """

        batch_size = observations.size(0)
        
        actions = torch.LongTensor(actions)
        onehot_actions = torch.zeros(batch_size, self.num_actions, (*observations.size())[1:])
        onehot_actions[range(batch_size), actions] = 1
        
        inputs = torch.cat([observations, onehot_actions], 1)
        
        x = F.relu( self.conv1( inputs) )
        x = F.relu( self.conv2( inputs) )
        x = x + F.relu( self.conv2(inputs) )

        output = self.output_conv( x)

        reward_x = F.relu( self.reward_conv1( x) )
        reward_x = F.relu( self.reward_conv2( reward_x) )
        reward_x = reward_x.view(batch_size, -1)
        reward = self.reward_fc( reward_x)

        return output, reward 

