from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.functional import kl_div
import torch.distributions as distributions

from torch.utils.tensorboard import SummaryWriter

summary_writer: SummaryWriter = None

def compute_loss(states: torch.Tensor,
                 pi_mcts: torch.FloatTensor,
                 values: torch.FloatTensor,
                 apprentice_model: nn.Module,
                 iteration_count: int) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x state_size: States visited by the agent.
    :param pi_mcts: Dimension: batch_size x number_actions,
                    Policy found by MCTS on each :param: states
    :param values: TODO
    :param apprentice_model: Neural network which imitates :param: target_action_distributions.

    :returns: Weighted loss between
              1 - Imitation learning loss (copying MCTS actions)
              2 - Value loss (estimating value of state)
              3 - Policy inference loss (imitating opponents)
    '''
    predictions = apprentice_model(states)

    # returns policy loss (cross entropy against normalized_child_visitations):

    # learning to copy expert: Cross entropy
    cross_entropy_policy_loss, kl_divergence = cross_entropy_loss(pi_mcts, predictions['probs'])

    # Learning game outcomes: Mean Square Error
    value_loss = nn.MSELoss()(values.view((-1, 1)), predictions['v'])

    # Learning to model opponents: Cross entropy loss
    opponent_modelling_loss = None  # TODO: cross_entropy_policy_loss between opponet targets and predictions

    total_loss = cross_entropy_policy_loss + value_loss

    # Sumary writer:
    # Policy inference (opponent modelling) loss
    # Policy inference weight
    if summary_writer is not None:
        summary_writer.add_scalar('Training/Policy_loss', cross_entropy_policy_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Value_loss', value_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Total_loss', total_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Kullback-Leibler_divergence', kl_divergence.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Apprentice_entropy', predictions['entropy'].mean().cpu().item(), iteration_count)
    return total_loss


def cross_entropy_loss(target: torch.Tensor,
                       model_predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    entropy_target = distributions.Categorical(probs=target).entropy()
    kl_divergence  = kl_div(model_predictions.log(), target, reduction='batchmean')
    return (entropy_target.mean() + kl_divergence), kl_divergence
