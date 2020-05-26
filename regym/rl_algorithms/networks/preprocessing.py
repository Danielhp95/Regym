from typing import List

import numpy as np
import torch


def turn_into_single_element_batch(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)


def flatten_and_turn_into_batch(x: List[np.ndarray]) -> torch.Tensor:
    return torch.tensor(
            [np.concatenate(x_i, axis=None)
             for x_i in x]).type(torch.FloatTensor)


def batch_vector_observation(x: List[np.ndarray]) -> torch.Tensor:
    return torch.stack([torch.tensor(x_i) for x_i in x]).type(torch.FloatTensor)
