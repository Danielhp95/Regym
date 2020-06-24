#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from typing import List, Callable, Iterable, Tuple
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from regym.rl_algorithms.networks.utils import layer_init, layer_init_lstm
from regym.rl_algorithms.networks.utils import convolutional_layer_output_dimensions, compute_convolutional_dimension_transforms
from regym.rl_algorithms.networks.utils import create_convolutional_layers


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


class Convolutional2DBody(nn.Module):
    def __init__(self, input_shape: Tuple[int, int],
                 channels: List[int], kernel_sizes: List[int],
                 paddings: List[int], strides: List[int],
                 residual_connections: List[Tuple[int, int]] = [],
                 use_batch_normalization=False,
                 gating_function: Callable = F.relu):
        '''
        :param input_shape: (Height x Width) dimensions of input tensors
        :param channels: List with number of channels for each convolution
        :param kernel_sizes: List of 'k' the size of the square kernel sizes for each convolution
        :param paddings: List with square paddings 'p' for each convolution
        :param strides: List with square stridings 's' for each convolution
        :param residual_connections: (l1, l2) tuples denoting that output
                                     from l1 should be added to input of l2
        :param use_batch_normalization: Whether to use BatchNorm2d after each convolution
        :param gating_function: Gating function to use after each convolution
        '''
        super().__init__()
        self.check_input_validity(channels, kernel_sizes, paddings, strides)
        self.gating_function = gating_function
        height_in, width_in = input_shape

        self.dimensions = compute_convolutional_dimension_transforms(
                height_in, width_in, channels, kernel_sizes, paddings, strides)

        convs = self.layer_connections(self.dimensions, residual_connections,
                                       i_in=0, i_max=(len(self.dimensions) - 1),  # check i_max is correct
                                       Cs=channels, Ks=kernel_sizes,
                                       Ps=paddings, Ss=strides,
                                       use_batch_normalization=use_batch_normalization)
        self.convolutions = nn.ModuleList(convs)

        output_height, output_width = self.dimensions[-1]
        self.feature_dim = output_height * output_width * channels[-1]

    def forward(self, x):
        conv_map = reduce(lambda acc, layer: self.gating_function(layer(acc)),
                          self.convolutions, x)
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

    def layer_connections(self, dimensions: List[Tuple[int, int]],
                          residual_connections: List[Tuple[int, int]],
                          i_in, i_max,
                          Cs, Ks, Ps, Ss,
                          use_batch_normalization) -> List[nn.Module]:
        if i_in == i_max: return []
        if residual_connections == []:
            return [create_convolutional_layers(Cs[i_in:], Ks[i_in:], Ps[i_in:],
                                                Ss[i_in:], use_batch_normalization)]
        l_in, l_out = residual_connections[0]
        if l_in == i_in:  # Start of residual block
            length = slice(l_in, l_out+1)
            res = ConvolutionalResidualBlock(dimensions[l_in],
                     Cs[length], Ks[length], Ps[length], Ss[length],
                     use_batch_normalization)
            return [res] + self.layer_connections(dimensions, residual_connections[1:],
                    l_out, i_max, Cs, Ks, Ps, Ss, use_batch_normalization)
        if l_in > i_in:  # Start of non-residual block
            length = slice(i_in, l_in + 1)
            con = create_convolutional_layers(Cs[length], Ks[length],
                                              Ps[length], Ss[length],
                                              use_batch_normalization)
            return [con] + self.layer_connections(dimensions, residual_connections,
                    l_in, i_max, Cs, Ks, Ps, Ss, use_batch_normalization)


class ConvolutionalResidualBlock(nn.Module):

    def __init__(self, input_shape: Tuple[int, int],
                 channels: List[int], kernel_sizes: List[int],
                 paddings: List[int], strides: List[int],
                 use_batch_normalization=False,
                 gating_function: Callable = F.relu):
        super().__init__()
        self.use_1x1conv = channels[0] != channels[-1]
        if self.use_1x1conv:
            self.residual_conv = nn.Conv2d(channels[0], channels[-1], kernel_size=1)

        height_in, width_in = input_shape
        self.dimensions = compute_convolutional_dimension_transforms(
                height_in, width_in, channels, kernel_sizes, paddings, strides)
        self.convolutions = create_convolutional_layers(
                channels, kernel_sizes, paddings, strides, use_batch_normalization)

        self.gating_function = gating_function
        output_height, output_width = self.dimensions[-1]
        self.feature_dim = output_height * output_width * channels[-1]

    def forward(self, x):
        x2 = reduce(lambda acc, layer: self.gating_function(layer(acc)),
                    self.convolutions, x)
        if self.use_1x1conv: x = self.residual_conv(x)
        return x + x2


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


class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x
