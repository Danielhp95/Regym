from typing import Tuple, List
import os

import numpy as np
import torch
import torch.nn as nn


class BaseNet:
    def __init__(self):
        # Sum a large negative constant to illegal action logits before taking the
        # max. This prevents illegal action values from being considered as target.
        self.ILLEGAL_ACTIONS_LOGIT_PENALTY = -1e9
        self.EPS = 1e-9

    def forward(x, legal_actions: List[int]):
        pass

    def _mask_ilegal_action_logits(self, logits: torch.Tensor,
                                   legal_actions: List[int], action_dim):
        '''
        TODO: document
        If legal_actions contains a singleton None list: [None]
        it means all actions are valid
        '''
        num_batches = logits.size()[0]
        is_multi_action = (num_batches > 1) or ((num_batches >= 1) and isinstance(legal_actions[0], list))
        if not is_multi_action:  # legal_actions is a vector, each entry is a legal action
            illegal_action_mask = torch.tensor([float(i not in legal_actions)
                                                for i in range(action_dim)]).to(logits.device)
        if is_multi_action:  # legal_actions is a matrix, where each row is a vector of legal actions
            illegal_action_mask = torch.tensor([[float(i not in legal_actions[j]) if legal_actions[j] else 0
                                                for i in range(action_dim)]
                                                for j in range(num_batches)]).to(logits.device)
        illegal_logit_penalties = illegal_action_mask * self.ILLEGAL_ACTIONS_LOGIT_PENALTY
        masked_logits = logits + illegal_logit_penalties
        return masked_logits


def compute_weights_decay_loss(model: torch.nn.Module, decay_rate: float = 1e-1) -> torch.Tensor:
    '''
    A form of regularization. Computes a loss aiming to regress
    network parameter values to 0.
    :param model: Model whose parameters will be used to compute the decay loss
    :param decay_rate: coefficient determining the magnitude (severity) of decay
    :returns: Weight decay loss
    '''
    return decay_rate * sum([torch.mean(param * param)
                             for param in model.parameters()])


def hard_update(fromm: torch.nn.Module, to: torch.nn.Module):
    '''
    Updates network parameters from :param fromm: to :param to:.
    Useful for updating target networks in DQN algorithms
    '''
    for fp, tp in zip(fromm.parameters(), to.parameters()):
        fp.data.copy_(tp.data)


def soft_update(fromm: torch.nn.Module, to: torch.nn.Module, tau: float):
    with torch.no_grad():
        for p, p_targ in zip(fromm.parameters(), to.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(tau)
            p_targ.data.add_((1 - tau) * p.data)


def layer_init(layer, w_scale=1.0) -> nn.Module:
    ''' Xavier init '''
    nn.init.orthogonal_(layer.weight.data)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.constant_(layer.bias.data, 0)
    return layer


def layer_init_lstm(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight_ih.data)
    nn.init.orthogonal_(layer.weight_hh.data)
    layer.weight_ih.data.mul_(w_scale)
    layer.weight_hh.data.mul_(w_scale)
    nn.init.constant_(layer.bias_ih.data, 0)
    nn.init.constant_(layer.bias_hh.data, 0)
    return layer


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = torch.tensor(x, dtype=torch.float32)
    return x


def random_seed(seed=None):
    np.random.seed(seed)
    torch.manual_seed(np.random.randint(int(1e6)))


def set_one_thread():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    torch.set_num_threads(1)


def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


def epsilon_greedy(epsilon, x):
    if len(x.shape) == 1:
        return np.random.randint(len(x)) if np.random.rand() < epsilon else np.argmax(x)
    elif len(x.shape) == 2:
        random_actions = np.random.randint(x.shape[1], size=x.shape[0])
        greedy_actions = np.argmax(x, axis=-1)
        dice = np.random.rand(x.shape[0])
        return np.where(dice < epsilon, random_actions, greedy_actions)


def sync_grad(target_network, src_network):
    for param, src_param in zip(target_network.parameters(), src_network.parameters()):
        param._grad = src_param.grad.clone()


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    remainder = len(indices) % batch_size
    if remainder:
        yield indices[-remainder:]


def convolutional_layer_output_dimensions(height: int, width: int,
                                          kernel_size: int, dilation: int, padding: int,
                                          stride: int) -> Tuple[int, int]:
    '''
    From https://pytorch.org/docs/stable/nn.html?highlight=torch%20nn%20conv2d#torch.nn.Conv2d
    '''
    height_out = 1 + ((height + 2 * padding - dilation * (kernel_size - 1) - 1) \
                      // stride)
    width_out = 1 + ((width + 2 * padding - dilation * (kernel_size - 1) - 1) \
                      // stride)
    return height_out, width_out


def compute_convolutional_dimension_transforms(height_in, width_in,
                                               channels, kernel_sizes, paddings,
                                               strides) -> List[Tuple[int, int]]:
    '''
    Computes the (height x width) dimension at each conv layer as a
    tensor of size=(:param: height_in, :param: width_in) passes through them
    '''
    dimensions = [(height_in, width_in)]
    dim_height, dim_width = height_in, width_in
    for c_in, c_out, k, p, s in zip(channels, channels[1:], kernel_sizes, paddings, strides):
        dim_height, dim_width = convolutional_layer_output_dimensions(dim_height, dim_width, k, 1, p, s)
        if dim_height < 1 or dim_width < 1:
            raise ValueError(f'At Convolutional layer {len(self.dimensions)} the dimensions of the convoluional map became invalid (less than 1): height = {dim_height}, width = {dim_width}')
        dimensions.append((dim_height, dim_width))
    return dimensions


def create_convolutional_layers(channels: List[int], kernel_sizes: List[int],
                                paddings: List[int], strides: List[int],
                                use_batch_normalization: bool) -> nn.Sequential:
    '''
    :param channels: List with number of channels for each convolution
    :param kernel_sizes: List of 'k' the size of the square kernel sizes for each convolution
    :param paddings: List with square paddings 'p' for each convolution
    :param strides: List with square stridings 's' for each convolution
    :param use_batch_normalization: Whether to use BatchNorm2d after each convolution
    '''
    convolutions = []
    for c_in, c_out, k, p, s in zip(channels, channels[1:], kernel_sizes, paddings, strides):
        convolutions += [layer_init(nn.Conv2d(in_channels=c_in, out_channels=c_out,
                                              kernel_size=k, stride=s, padding=p))]

        if use_batch_normalization: convolutions += [nn.BatchNorm2d(c_out)]
    return nn.Sequential(*convolutions)
