import os
import numpy as np
import torch
import torch.nn as nn


class BaseNet:
    def __init__(self):
        # Sum a large negative constant to illegal action logits before taking the
        # max. This prevents illegal action values from being considered as target.
        self.ILLEGAL_ACTIONS_LOGIT_PENALTY = -1e9


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
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
    x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
    return x


def range_tensor(end):
    return torch.arange(end).long().to(Config.DEVICE)


def to_np(t):
    return t.cpu().detach().numpy()


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


class PreprocessFunction():

    def __init__(self, state_space_size, use_cuda=False):
        self.state_space_size = state_space_size
        self.use_cuda = use_cuda

    def __call__(self, x):
        x = np.concatenate(x, axis=None)
        if self.use_cuda:
            return torch.from_numpy(x).unsqueeze(0).type(torch.cuda.FloatTensor)
        else:
            return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)


def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    remainder = len(indices) % batch_size
    if remainder:
        yield indices[-remainder:]
