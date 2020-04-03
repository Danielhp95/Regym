#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from typing import List, Callable, Iterable
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from regym.rl_algorithms.networks.utils import layer_init, layer_init_lstm
from regym.rl_algorithms.networks.utils import convolutional_layer_output_dimensions



class SequentialBody(nn.Module):
    '''
    A wrapper around torch.nn.Sequential so that it exposes property 'feature_dim'
    '''

    def __init__(self, *bodies: Iterable[nn.Module]):
        super().__init__()
        self.sequence = nn.Sequential(*bodies)
        self.feature_dim = self.sequence[-1].feature_dim

    def forward(self, x: torch.Tensor):
        return self.sequence(x)


class NatureConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


class Convolutional2DBody(nn.Module):
    def __init__(self, input_shape: List[int],
                 channels: List[int], kernel_sizes: List[int],
                 paddings: List[int], strides: List[int],
                 use_batch_normalization=False,
                 gating_function: Callable = F.relu):
        '''
        TODO
        '''
        super().__init__()

        self.check_input_validity(channels, kernel_sizes,
                                  paddings, strides)

        self.gating_function = gating_function
        height_in, width_in = input_shape

        self.convolutions, (dim_height, dim_width) = \
                self.create_convolutional_layers(height_in, width_in, channels,
                                                 kernel_sizes, paddings, strides,
                                                 use_batch_normalization=use_batch_normalization)
        # TODO: get feature_dim should be set here
        self.feature_dim = dim_height * dim_width * channels[-1]

    def create_convolutional_layers(self, height_in: int, width_in: int,
                                    channels: List[int], kernel_sizes: List[int],
                                    paddings: List[int],
                                    strides: List[int],
                                    use_batch_normalization: bool) -> nn.ModuleList:
        # We compute the dimension of the input. Useful for debuggin too
        self.dimensions = [(height_in, width_in)]

        dim_height, dim_width = height_in, width_in

        # Create convolutions and re-compute dimensions as
        # new conv layers are added
        convolutions = nn.ModuleList()
        for c_in, c_out, k, p, s in zip(
                channels, channels[1:], kernel_sizes, paddings, strides):
            convolutions.append(
                    layer_init(
                        nn.Conv2d(in_channels=c_in, out_channels=c_out,
                                  kernel_size=k, stride=s, padding=p))
                    )

            if use_batch_normalization: convolutions.append(nn.BatchNorm2d(c_out))

            dim_height, dim_width = convolutional_layer_output_dimensions(
                    height=dim_height, width=dim_width, kernel_size=k,
                    dilation=1,  # TODO: support dilation, if it ever becomes useful.
                    padding=p, stride=s)
            if dim_height < 1 or dim_width < 1:
                raise ValueError(f'At Convolutional layer {len(self.dimensions)} the dimensions of the convoluional map became invalid (less than 1): height = {dim_height}, width = {dim_width}')
            self.dimensions.append((dim_height, dim_width))

        return convolutions, (dim_height, dim_width)

    def forward(self, x):
        conv_map = reduce(lambda acc, layer: layer(acc), self.convolutions, x)
        # Without start_dim, we are flattening over the entire batch!
        flattened_conv_map = conv_map.flatten(start_dim=1)
        flat_embedding = self.gating_function(flattened_conv_map)
        return flat_embedding

    def check_input_validity(self, channels, kernel_sizes, paddings, strides):
        if len(channels) < 2: raise ValueError('At least 2 channels must be specified')
        if len(kernel_sizes) != (len(channels) - 1):
            raise ValueError(f'{len(kernel_sizes)} kernel_sizes were specified, but exactly {len(channels) -1} are required')
        if len(kernel_sizes) != (len(channels) - 1):
            raise ValueError(f'{len(kernel_sizes)} kernel_sizes were specified, but exactly {len(channels) -1} are required')
        if len(paddings) != (len(channels) - 1):
            raise ValueError(f'{len(paddings)} paddings were specified, but exactly {len(channels) -1} are required')
        if len(strides) != (len(channels) - 1):
            raise ValueError(f'{len(strides)} strides were specified, but exactly {len(channels) -1} are required')


class DDPGConvBody(nn.Module):
    def __init__(self, in_channels=4):
        super(DDPGConvBody, self).__init__()
        self.feature_dim = 39 * 39 * 32
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

    def forward(self, x):
        y = F.elu(self.conv1(x))
        y = F.elu(self.conv2(y))
        y = y.view(y.size(0), -1)
        return y


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        self.layers = nn.ModuleList([layer_init(nn.Linear(dim_in, dim_out))
                                     for dim_in, dim_out in zip(dims[:-1],
                                                                dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        return reduce(lambda acc, layer: self.gate(layer(acc)), self.layers, x)


class LSTMBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(256), gate=F.relu):
        super(LSTMBody, self).__init__()
        dims = (state_dim, ) + hidden_units
        # Consider future cases where we may not want to initialize the LSTMCell(s)
        self.layers = nn.ModuleList([layer_init_lstm(nn.LSTMCell(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.feature_dim = dims[-1]
        self.gate = gate

    def forward(self, x):
        '''
        :param x: input to LSTM cells. Structured as (input, (hidden_states, cell_states)).
        hidden_states: list of hidden_state(s) one for each self.layers.
        cell_states: list of hidden_state(s) one for each self.layers.
        '''
        x, (hidden_states, cell_states) = x
        next_hstates, next_cstates = [], []
        for idx, (layer, hx, cx) in enumerate(zip(self.layers, hidden_states, cell_states) ):
            batch_size = x.size(0)
            if hx.size(0) == 1: #then we have just resetted the values, we need to expand those:
                hx = torch.cat( [hx]*batch_size, dim=0)
                cx = torch.cat( [cx]*batch_size, dim=0)
            elif hx.size(0) != batch_size:
                raise NotImplemented("Sizes of the hidden states and the inputs do not coincide.")

            nhx, ncx = layer(x, (hx, cx) )
            next_hstates.append(nhx)
            next_cstates.append(ncx)
            # Consider not applying activation functions on last layer's output
            if self.gate is not None:
                x = self.gate(nhx)

        return x, (next_hstates, next_cstates)

    def get_reset_states(self, cuda=False):
        hidden_states, cell_states = [], []
        for layer in self.layers:
            h = torch.zeros(1,layer.hidden_size)
            if cuda:
                h = h.cuda()
            hidden_states.append(h)
            cell_states.append(h)
        return (hidden_states, cell_states)

class TwoLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units=(64, 64), gate=F.relu):
        super(TwoLayerFCBodyWithAction, self).__init__()
        hidden_size1, hidden_size2 = hidden_units
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_size1))
        self.fc2 = layer_init(nn.Linear(hidden_size1 + action_dim, hidden_size2))
        self.gate = gate
        self.feature_dim = hidden_size2

    def forward(self, x, action):
        x = self.gate(self.fc1(x))
        phi = self.gate(self.fc2(torch.cat([x, action], dim=1)))
        return phi


class OneLayerFCBodyWithAction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_units, gate=F.relu):
        super(OneLayerFCBodyWithAction, self).__init__()
        self.fc_s = layer_init(nn.Linear(state_dim, hidden_units))
        self.fc_a = layer_init(nn.Linear(action_dim, hidden_units))
        self.gate = gate
        self.feature_dim = hidden_units * 2

    def forward(self, x, action):
        phi = self.gate(torch.cat([self.fc_s(x), self.fc_a(action)], dim=1))
        return phi


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
