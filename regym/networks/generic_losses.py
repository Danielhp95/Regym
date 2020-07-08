from typing import Tuple

import torch
import torch.distributions as distributions
from torch.nn.functional import kl_div


def cross_entropy_loss(model_prediction: torch.Tensor,
                       target: torch.Tensor) \
                               -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    :param target: Tensor containing the target distribution.
                   
    :param model_prediction: Tensor containing the model's predctions.
                             Expected size: [batch_size, individual_prediction_size]
    :returns: Cross entropy loss between target and model prediction, and KL divergence
    '''
    # TODO: remove KL
    epsilon = 1e-4
    # To prevent log(0) from exploding to infinity
    safe_target = target.clamp(min=1e-8)
    return -1. * torch.sum(model_prediction * safe_target.log(), dim=1).mean(), None
    #entropy_target = distributions.Categorical(probs=target).entropy()
    #kl_divergence  = kl_div(model_predictions.log(), target, reduction='batchmean')
    #return (entropy_target.mean() + kl_divergence), kl_divergence
