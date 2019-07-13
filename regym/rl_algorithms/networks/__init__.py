from .networks import hard_update, soft_update
from .networks import LeakyReLU
from .networks import DQN, DuelingDQN
from .networks import ActorNN, CriticNN
from .ppo_network_heads import CategoricalActorCriticNet, GaussianActorCriticNet
from .ppo_network_bodies import FCBody, LSTMBody, GRUBody, ConvolutionalBody, ConvolutionalLstmBody, ConvolutionalGruBody
from .utils import PreprocessFunction, CNNPreprocessFunction, ResizeCNNPreprocessFunction
from .utils import random_sample

import torch.nn.functional as F 

def choose_architecture( architecture,input_dim=None, hidden_units_list=None,
                        input_shape=None,feature_dim=None, nbr_channels_list=None, kernels=None, strides=None, paddings=None):
    if 'LSTM-RNN' in architecture:
        return LSTMBody(input_dim, hidden_units=hidden_units_list, gate=F.leaky_relu)
    if 'GRU-RNN' in architecture:
        return GRUBody(input_dim, hidden_units=hidden_units_list, gate=F.leaky_relu)
    if architecture == 'MLP':
        return FCBody(input_dim, hidden_units=hidden_units_list, gate=F.leaky_relu)
    
    if architecture == 'CNN':
        channels = [input_shape[0]] + nbr_channels_list
        phi_body = ConvolutionalBody(input_shape=input_shape,
                                     feature_dim=feature_dim,
                                     channels=channels,
                                     kernel_sizes=kernels,
                                     strides=strides,
                                     paddings=paddings)
    if architecture == 'CNN-RNN':
        channels = [input_shape[0]] + nbr_channels_list
        phi_body = ConvolutionalLstmBody(input_shape=input_shape,
                                     feature_dim=feature_dim,
                                     channels=channels,
                                     kernel_sizes=kernels,
                                     strides=strides,
                                     paddings=paddings,
                                     hidden_units=hidden_units_list)
    return phi_body