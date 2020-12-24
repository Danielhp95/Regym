from typing import List, Union, Callable

import numpy as np
import torch


def turn_into_single_element_batch(x: np.ndarray, use_cuda=False) -> torch.Tensor:
    tensor = torch.Tensor(x).unsqueeze(0).type(torch.FloatTensor)
    if use_cuda: tensor = tensor.cuda()
    return tensor


def flatten_and_turn_into_single_element_batch(x: np.ndarray,
                                               use_cuda=False) -> torch.Tensor:
    return turn_into_single_element_batch(np.concatenate(x, axis=None))


def flatten_and_turn_into_batch(x: List[np.ndarray]) -> torch.Tensor:
    return torch.tensor(
            [np.concatenate(x_i, axis=None)
             for x_i in x]).type(torch.FloatTensor)


def batch_vector_observation(x: List[Union[np.ndarray, List]],
                             type: str = 'float64') -> torch.Tensor:
    return torch.stack([
        torch.from_numpy(x_i.astype(type)) if isinstance(x_i, np.ndarray) else torch.tensor(x_i)
        for x_i in x]).type(torch.FloatTensor)


def flatten_last_dim_and_batch_vector_observation(x: List[Union[np.ndarray, List]],
                                                  type: str = 'float64') \
                                                  -> torch.Tensor:
    r'''
    First, it batches a vector observation as in `batch_vector_observation` fn above
    Second, it flattens the first (non batch) dimension of tensor into the
    second (non batch) dimension such that if :param: x is of shape
    (n, m, p), the resulting tensor will be of shape (n \times m, p).

    :param x: Input to be processed into a tensor
    :param type: np.dtype that will be used to internally cast :param x:.
                 NOTE: output tensor will be of type torch.FloatTensor.
    :returns: FloatTensor processed as explained above.
    '''
    batched_tensor = batch_vector_observation(x, type=type)
    # Dimensions: [batch dim, dim 1, dim 2]
    if len(batched_tensor.shape) < 3:
        raise ValueError('Input tensor must have at least 2 dimensions '
                         'besides batch dimension')
    return batched_tensor.flatten(start_dim=1, end_dim=2)


def keep_last_stack_and_batch_vector_observation(x: List[Union[np.ndarray, List]]) \
                                                                 -> torch.Tensor:
    '''
    Useful to use agents on environments with framestack when these agents were trained
    without framestack
    '''
    batched_tensor = batch_vector_observation(x)
    num_stacks = batched_tensor.shape[1]
    tensor_with_only_one_stack = batched_tensor[:, -1]
    # Dim 0 -> batch dimension. Dim 1 -> stack dimension. Dim 2.. -> obs dimension
    return tensor_with_only_one_stack.squeeze(dim=1)


def parse_preprocessing_fn(fn_name: str) -> Callable:
    '''
    Parses :param: fn_name to see if there is a preprocessing function
    with the same name. Useful when one wants to define a preprocessing
    function in a config file
    '''
    if fn_name == 'turn_into_single_element_batch':
        return turn_into_single_element_batch
    if fn_name == 'flatten_and_turn_into_batch':
        return flatten_and_turn_into_batch
    if fn_name == 'batch_vector_observation':
        return batch_vector_observation
    if fn_name == 'flatten_last_dim_and_batch_vector_observation':
        return flatten_last_dim_and_batch_vector_observation
    if fn_name == 'keep_last_stack_and_batch_vector_observation':
        return keep_last_stack_and_batch_vector_observation
    else:
        raise ValueError('Couldd not parse preprocessing function '
                         f'from name {fn_name}')
