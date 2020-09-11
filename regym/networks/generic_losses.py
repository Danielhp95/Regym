from typing import Tuple

import torch


def cross_entropy_loss(model_prediction: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    '''
    Computes cross entropy loss of :param: model_prediction
    against :param: target

    :param model_prediction: Tensor containing the model's predctions.
                             Expected size: [batch_size, individual_prediction_size]
    :param target: Tensor containing the target distribution.
    :returns: Cross entropy loss between target and model prediction
    '''
    safe_target = target.clamp(min=1e-8)  # To prevent log(0) from exploding
    return -1. * (model_prediction * safe_target.log()).sum(dim=1).mean()
