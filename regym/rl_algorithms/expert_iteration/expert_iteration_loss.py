from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from regym.networks.generic_losses import cross_entropy_loss

summary_writer: SummaryWriter = None

def compute_loss(states: torch.FloatTensor,
                 pi_mcts: torch.FloatTensor,
                 values: torch.FloatTensor,
                 opponent_policy: Optional[torch.FloatTensor],
                 opponent_s: Optional[torch.FloatTensor],
                 use_agent_modelling: bool,
                 apprentice_model: nn.Module,
                 iteration_count: int) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x state_size: States visited by the agent.
    :param pi_mcts: Dimension: batch_size x number_actions,
                    Policy found by MCTS on each :param: states
    :param values: state-value for each :param state. Dimension: batch_size x 1
    :param opponent_policy: Dimension: batch_size x number_actions
    :param use_agent_modelling: Whether to include opponent modelling loss
    :param apprentice_model: Neural network which imitates :param: target_action_distributions.

    :returns: Weighted loss between
              1 - Imitation learning loss (copying MCTS actions)
              2 - Value loss (estimating value of state)
    Optional  3 - Policy inference (opponent modelling) loss (imitating opponents)
    '''
    predictions = apprentice_model(states)

    # returns policy loss (cross entropy against normalized_child_visitations):

    # learning to copy expert: Cross entropy
    policy_imitation_loss = cross_entropy_loss(predictions['probs'], pi_mcts)

    # For logging purposes
    kl_divergence = torch.nn.functional.kl_div(predictions['probs'], pi_mcts.log(), reduction='batchmean')

    # Learning game outcomes: Mean Square Error
    value_loss = nn.MSELoss()(values.view((-1, 1)), predictions['V'])

    # Vanilla Expert Iteration loss
    exit_loss = policy_imitation_loss + value_loss

    # Learning to model opponents: Cross entropy loss
    if not use_agent_modelling:
        total_loss = exit_loss
    else:
        opponent_modelling_loss = compute_opponent_modelling_loss(
            opponent_policy,
            opponent_s, apprentice_model
        )

        # dynamically computed weight
        policy_inference_weight = 1 / (torch.sqrt(opponent_modelling_loss))
        '''First focus on learning the opponent, then focus on baseline loss'''

        total_loss = exit_loss * policy_inference_weight + opponent_modelling_loss

    # Sumary writer:
    # Policy inference (opponent modelling) loss
    # Policy inference weight
    if summary_writer is not None:
        summary_writer.add_scalar('Training/Policy_loss', policy_imitation_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Value_loss', value_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Expert_Iteration_loss', exit_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Total_loss', total_loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Kullback-Leibler_divergence', kl_divergence.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Apprentice_entropy', predictions['entropy'].mean().cpu().item(), iteration_count)
        if use_agent_modelling:
            summary_writer.add_scalar('Training/Opponent_modelling_loss', opponent_modelling_loss.cpu().item(), iteration_count)
            summary_writer.add_scalar('Training/Policy_inference_weight', opponent_modelling_loss.cpu().item(), iteration_count)
    return total_loss


def compute_opponent_modelling_loss(opponent_policy: torch.Tensor,
                                    opponent_s: torch.Tensor,
                                    apprentice_model: nn.Module) \
                                    -> torch.Tensor:
    filtered_opponent_policies, filtered_opponent_s = filter_nan_tensors(
        opponent_policy,
        opponent_s)
    opponent_predictions = apprentice_model(filtered_opponent_s)

    opponent_modelling_loss = cross_entropy_loss(
        model_prediction=opponent_predictions['policy_0']['probs'],
        target=filtered_opponent_policies)
    return opponent_modelling_loss


def filter_nan_tensors(opponent_policy, opponent_s: torch.Tensor) \
                       -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Note: Preprocessing:
    Because we deal with sequential games, when this agent's actions finish
    the episode, the experience propagated to the agent has no extra_info regarding
    other agents, and a placeholder 'nan' value takes it's place which needs to be removed
    '''
    non_nan_indexes = ~(torch.any(torch.isnan(opponent_policy), dim=1))
    filtered_opponent_policies = opponent_policy[non_nan_indexes]
    filtered_opponent_s = opponent_s[non_nan_indexes]
    return filtered_opponent_policies, filtered_opponent_s
