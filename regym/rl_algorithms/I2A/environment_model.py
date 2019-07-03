import torch
from torch import nn
import torch.nn.functional as F


class EnvironmentModel(nn.Module):
    def __init__(self, observation_shape, num_actions, reward_size, conv_dim=32, use_cuda=False):
        """
        :param observation_shape: shape depth x height x width.
        :param num_actions: number of actions that are available in the environment.
        :param reward_size: number of dimensions of the reward vector. 
                            Eventhough OpenAI Gym Interface always provides
                            scalar reward function, it might be interesting 
                            in some other environments to predict a vector of
                            reward functions.
        :param conv_dim: number of convolution kernels to use for each conv layer.
        """
        super(EnvironmentModel, self).__init__()

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.reward_size = reward_size
        self.conv_dim = conv_dim
        self.use_cuda = use_cuda

        height_dim = self.observation_shape[1]

        # 8x8, conv_dim, /8 convolutional layer:
        self.conv1, height_dim = self._build_conv_layer(height_dim, 
                                                        in_channels=self.observation_shape[0]+self.num_actions,
                                                        out_channels=self.conv_dim,
                                                        k=8, stride=8, pad=0)    
        # two size-preserving convolutional layers:
        self.conv2, height_dim = self._build_conv_layer(height_dim, 
                                                        in_channels=self.conv_dim,
                                                        out_channels=self.conv_dim,
                                                        k=3, stride=1, pad=1)    
        self.conv3, height_dim = self._build_conv_layer(height_dim, 
                                                        in_channels=self.conv_dim,
                                                        out_channels=self.conv_dim,
                                                        k=3, stride=1, pad=1)    
        
        k=8
        stride=8
        pad=0
        dilation=1
        self.output_conv = nn.ConvTranspose2d(in_channels=self.conv_dim, out_channels=self.observation_shape[0],
            kernel_size=k, stride=stride, padding=pad,dilation=dilation)
        output_height_dim = (height_dim-1)*stride-2*pad+dilation*(k-1)+1
        
        k=3
        stride=1
        pad=0
        self.reward_conv1, reward_dim = self._build_conv_maxpool_layer(height_dim, 
                                                                        in_channels=self.conv_dim,
                                                                        out_channels=self.conv_dim,
                                                                        k=3, stride=1, pad=0)    
        self.reward_conv2, reward_dim = self._build_conv_maxpool_layer(reward_dim, 
                                                                        in_channels=self.conv_dim,
                                                                        out_channels=self.conv_dim,
                                                                        k=3, stride=1, pad=0)    
        
        self.reward_fc = nn.Linear(reward_dim*reward_dim*self.conv_dim, self.reward_size)

        if self.use_cuda: self = self.cuda()

    def _build_conv_layer(self, height_dim, in_channels, out_channels, k, stride, pad):
        new_height_dim = (height_dim-k+2*pad)//stride +1
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=k, stride=stride, padding=pad)
        return conv, new_height_dim

    def _build_conv_maxpool_layer(self, height_dim, in_channels, out_channels, k, stride, pad):
        new_height_dim = (height_dim-k+2*pad)//stride +1
        new_height_dim = (new_height_dim-2)//1 +1
        conv_maxpool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=k, stride=stride, padding=pad),
            nn.MaxPool2d(kernel_size=2,stride=1)
            )
        return conv_maxpool, new_height_dim

    def forward(self, observations, actions):
        """
        Returns the next observation and reward, given :param observations: 
        and :param actions:.
        :param observations: observation tensor of shape batch x depth x height x width.
        :param actions: action indexes tensor of shape batch x 1.
        """

        batch_size = observations.size(0)
        
        actions = actions.long()
        if self.use_cuda: actions = actions.cuda()

        onehot_actions = torch.zeros(batch_size, self.num_actions, *(observations.size())[2:])
        if self.use_cuda: onehot_actions = onehot_actions.cuda()

        onehot_actions[range(batch_size), actions] = 1
        inputs = torch.cat([observations, onehot_actions], 1)
        
        x = F.relu( self.conv1( inputs) )
        x = F.relu( self.conv2( x) )
        x = x + F.relu( self.conv2(x) )

        output = self.output_conv( x)

        reward_x = F.relu( self.reward_conv1( x) )
        reward_x = F.relu( self.reward_conv2( reward_x) )
        reward_x = reward_x.view(batch_size, -1)
        reward = self.reward_fc( reward_x)

        return output, reward 

