from typing import Tuple

import torch


def cross_entropy_loss(model_prediction: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    '''
    Further read: https://en.wikipedia.org/wiki/Cross_entropy

    Computes cross entropy loss of :param: model_prediction
    against :param: target

    :param model_prediction: Tensor containing the model's predctions.
                             Expected size: [batch_size, individual_prediction_size]
    :param target: Tensor containing the target distribution.
    :returns: Cross entropy loss between target and model prediction
    '''
    safe_predictions = model_prediction.clamp(min=1e-8)  # To prevent log(0) from exploding
    return -1. * (target * safe_predictions.log()).sum(dim=1).mean()
