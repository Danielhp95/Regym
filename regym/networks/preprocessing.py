from typing import List, Union

import numpy as np
import torch


def turn_into_single_element_batch(x: np.ndarray, use_cuda=False) -> torch.Tensor:
    x = np.concatenate(x, axis=None) 
    tensor = torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)
    if use_cuda: tensor = tensor.cuda()
    return tensor


def flatten_and_turn_into_batch(x: List[np.ndarray]) -> torch.Tensor:
    return torch.tensor(
            [np.concatenate(x_i, axis=None)
             for x_i in x]).type(torch.FloatTensor)


def batch_vector_observation(x: List[Union[np.ndarray, List]],
                             type: str = 'float64') -> torch.Tensor:
    return torch.stack([
        torch.from_numpy(x_i.astype(type)) if isinstance(x_i, np.ndarray) else torch.tensor(x_i)
        for x_i in x]).type(torch.FloatTensor)
