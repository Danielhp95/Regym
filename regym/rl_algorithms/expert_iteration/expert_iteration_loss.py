from typing import Dict, List

import torch
import torch.nn as nn

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
    
    # cross entropy
    policy_loss = -torch.sum(pi_mcts * torch.log(predictions['probs'])) / pi_mcts.size()[0]
    value_loss = nn.MSELoss()(values, predictions['v'])

    # Opponent modelling loss (cross entropy loss)

    # Sumary writer:
    # Value loss
    # policy loss
    # Policy inference (opponent modelling) loss
    # Policy inference weight
    # Total loss
    total_loss = policy_loss + value_loss
    if summary_writer is not None:
        summary_writer.add_scalar('Training/Policy_loss', policy_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Value_loss', value_loss.cpu().item(), iteration_count)
    return total_loss
