import torch
import torch.autograd
import torchvision.transforms as T
import numpy as np


def PreprocessFunctionConcatenate(x, use_cuda=False):
    x = np.concatenate(x, axis=None)
    if use_cuda:
        return torch.from_numpy(x).unsqueeze(0).type(torch.cuda.FloatTensor)
    return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)

def PreprocessFunction(x, use_cuda=False):
    if use_cuda:
        return torch.from_numpy(x).type(torch.cuda.FloatTensor)
    else:
        return torch.from_numpy(x).type(torch.FloatTensor)

def ResizeCNNPreprocessFunction(x, size, use_cuda=False, normalize_rgb_values=True):
    '''
    Used to resize, normalize and convert OpenAI Gym raw pixel observations,
    which are structured as numpy arrays of shape (Height, Width, Channels),
    into the equivalent Pytorch Convention of (Channels, Height, Width).
    Required for torch.nn.Modules which use a convolutional architechture.

    :param x: Numpy array to be processed
    :param size: int or tuple, (height,width) size
    :param use_cuda: Boolean to determine whether to create Cuda Tensor
    :param normalize_rgb_values: Maps the 0-255 values of rgb colours
                                 to interval (0-1)
    '''
    if isinstance(size, int): size = (size,size)
    scaling_operation = T.Compose([T.ToPILImage(),
                                    T.Resize(size=size)])
    if len(x.shape) == 4:
        #batch:
        imgs = []
        for i in range(x.shape[0]):
            img = np.array(scaling_operation(x[i]))
            img = torch.from_numpy(img.transpose((2, 0, 1))).unsqueeze(0)
            imgs.append(img)
        x = torch.cat(imgs, dim=0)
    else:
        x = scaling_operation(x)
        x = np.array(x)
        x = x.transpose((2, 0, 1))
        x = torch.from_numpy(x).unsqueeze(0)
    # WATCHOUT: it is necessary to cast the tensor into float before doing
    # the division, otherwise the result is yielded as a uint8 (full of zeros...)
    x = x.type(torch.FloatTensor) / 255. if normalize_rgb_values else x
    if use_cuda:
        return x.type(torch.cuda.FloatTensor)
    return x.type(torch.FloatTensor)


def CNNPreprocessFunction(x, use_cuda=False, normalize_rgb_values=True):
    '''
    Used to normalize and convert OpenAI Gym raw pixel observations,
    which are structured as numpy arrays of shape (Height, Width, Channels),
    into the equivalent Pytorch Convention of (Channels, Height, Width).
    Required for torch.nn.Modules which use a convolutional architechture.

    :param x: Numpy array to be processed
    :param use_cuda: Boolean to determine whether to create Cuda Tensor
    :param normalize_rgb_values: Maps the 0-255 values of rgb colours
                                 to interval (0-1)
    '''
    x = x.transpose((2, 0, 1))
    x = x / 255. if normalize_rgb_values else x
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
