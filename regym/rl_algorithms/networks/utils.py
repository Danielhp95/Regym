import torch
import torch.autograd
import numpy as np


def PreprocessFunction(x, use_cuda=False):
    x = np.concatenate(x, axis=None)
    if use_cuda:
        return torch.from_numpy(x).unsqueeze(0).type(torch.cuda.FloatTensor)
    return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)


def CNNPreprocessFunction(x, use_cuda=False):
    '''
    Used to convert OpenAI Gym raw pixel observations,
    which are structured as numpy arrays of shape (Height, Width, Channels),
    into the equivalent Pytorch Convention of (Channels, Height, Width).
    Required for torch.nn.Modules which use a convolutional architechture.
    '''
    x = x.transpose((2, 0, 1))
    if use_cuda:
        return torch.from_numpy(x).unsqueeze(0).type(torch.cuda.FloatTensor)
    return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)


def random_sample(indices, batch_size):
    '''
    TODO
    :param indices:
    :param batch_size:
    :returns: Generator
    '''
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    remainder = len(indices) % batch_size
    if remainder:
        yield indices[-remainder:]


# Note, consider alternative way of calculating output size.
# Idea: use model.named_modules() generator to find last module and look at its
# Number of features / output channels (if the last module is a ConvNet)
def output_size_for_model(model, input_shape):
    '''
    Computes the size of the last layer of the :param model:
    which takes an input of shape :param input_shape:.

    :param model: torch.nn.Module model which takes an input of shape :param input_shape:
    :param input_shape: Shape of the input of the
    :returns: size of the flattened output torch.Tensor
    '''
    return model(torch.autograd.Variable(torch.zeros(1, *input_shape))).view(1, -1).size(1)
