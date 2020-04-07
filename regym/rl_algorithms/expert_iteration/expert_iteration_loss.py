from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn.functional import kl_div
import torch.distributions as distributions

from torch.utils.tensorboard import SummaryWriter

summary_writer = None

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

    # learning to copy expert: cross entropy
    entropy_pi_mcts = distributions.Categorical(probs=pi_mcts).entropy()
    kl_divergence_mcts_apprentice = kl_div(predictions['probs'].log(), pi_mcts,
                                           reduction='batchmean')
    cross_entropy_policy_loss = entropy_pi_mcts.mean() + kl_divergence_mcts_apprentice

    # Learning game outcomes: Mean Square Error
    value_loss = nn.MSELoss()(values, predictions['v'])

    total_loss = cross_entropy_policy_loss + value_loss

    # Opponent modelling loss (cross entropy loss)

    # Sumary writer:
    # Policy inference (opponent modelling) loss
    # Policy inference weight
    if summary_writer is not None:
        summary_writer.add_scalar('Training/Policy_loss', cross_entropy_policy_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Value_loss', value_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Total_loss', total_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Kullback-Leibler_divergence', kl_divergence_mcts_apprentice.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Apprentice_entropy', predictions['entropy'].mean().cpu().item(), iteration_count)
    return total_loss
