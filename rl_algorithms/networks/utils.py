import torch
import numpy as np


class PreprocessFunction(object):

    def __init__(self, state_space_size, use_cuda=False):
        self.state_space_size = state_space_size
        self.use_cuda = use_cuda

    def __call__(self, x):
        x = np.concatenate(x, axis=None)
        if self.use_cuda:
            return torch.from_numpy(x).unsqueeze(0).type(torch.cuda.FloatTensor)
        else:
            return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)
