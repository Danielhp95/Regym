from typing import Dict, List

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


# NOTE: rnn_states might be tricky to get working here
def compute_loss(states: torch.Tensor,
                 pi_mcts: torch.FloatTensor,
                 values: torch.FloatTensor,
                 apprentice_model: nn.Module) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x state_size: States visited by the agent.
    :param pi_mcts: Dimension: batch_size x number_actions,
                    Policy found by MCTS on each :param: states
    :param values: TODO
    :param apprentice_model: Neural network which imitates :param: target_action_distributions. 
    '''
    predictions = apprentice_model(states)

    import ipdb; ipdb.set_trace()
    # returns policy loss (cross entropy against normalized_child_visitations):
    
    # policy_loss = nn.CrossEntropyLoss()(pi_mcts, predictions['probs'])

    # value loss ()?
    value_loss = nn.MSELoss()(values, predictions['v'])

    # Opponent modelling loss (cross entropy loss)

    # Sumary writer:
    # Value loss
    # policy loss
    # Policy inference (opponent modelling) loss
    # Policy inference weight
    # Total loss
    total
    return policy_loss
