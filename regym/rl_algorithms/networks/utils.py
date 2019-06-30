import torch
import numpy as np
import torch.autograd


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


def output_size_for_model(model, input_shape):
    '''
    Computes the size of the last layer of the :param model:
    which takes an input of shape :param input_shape:.

    :param model: torch.nn.Module model which takes an input of shape :param input_shape:
    :param input_shape: Shape of the input of the
    :returns: size of the flattened output torch.Tensor
    '''
    return model(torch.autograd.Variable(torch.zeros(1, *input_shape))).view(1, -1).size(1)
