from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from regym.networks.generic_losses import cross_entropy_loss
from regym.networks.utils import entropy


def compute_loss(states: torch.FloatTensor,
                 pi_mcts: torch.FloatTensor,
                 values: torch.FloatTensor,
                 opponent_policy: Optional[torch.FloatTensor],
                 opponent_s: Optional[torch.FloatTensor],
                 use_agent_modelling: bool,
                 apprentice_model: nn.Module,
                 iteration_count: int,
                 summary_writer: SummaryWriter) -> torch.Tensor:
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
    policy_imitation_loss = cross_entropy_loss(
        model_prediction=predictions['probs'],
        target=pi_mcts
    )

    # For logging purposes
    kl_divergence = torch.nn.functional.kl_div(predictions['probs'].log(),
                                               pi_mcts,
                                               reduction='batchmean')

    # Learning game outcomes: Mean Square Error
    value_loss = nn.MSELoss()(values.view((-1, 1)), predictions['V'])

    # Vanilla Expert Iteration loss
    exit_loss = policy_imitation_loss + value_loss

    # Learning to model opponents: Cross entropy loss
    if not use_agent_modelling:
        total_loss = exit_loss
    else:
        (opponent_modelling_loss,
         policy_inference_weight,
         kl_divergence_opponent_modelling) = compute_opponent_modelling_loss(
            opponent_policy,
            opponent_s, apprentice_model
        )

        '''First focus on learning the opponent, then focus on baseline loss'''
        total_loss = exit_loss * policy_inference_weight + opponent_modelling_loss
        if summary_writer and (iteration_count % 10 == 0):
            log_opponent_modelling_loss_progress(summary_writer,
                                                 opponent_modelling_loss,
                                                 policy_inference_weight,
                                                 kl_divergence_opponent_modelling,
                                                 iteration_count)

    if summary_writer and (iteration_count % 10 == 0):
        log_exit_loss_progress(summary_writer,
                               policy_imitation_loss,
                               value_loss,
                               exit_loss,
                               total_loss,
                               kl_divergence,
                               values,
                               predictions,
                               pi_mcts,
                               iteration_count)
    return total_loss


def log_exit_loss_progress(summary_writer,
                           policy_imitation_loss,
                           value_loss,
                           exit_loss,
                           total_loss,
                           kl_divergence,
                           values,
                           predictions,
                           pi_mcts,
                           iteration_count):
    summary_writer.add_scalar('Training/Expert_policy_imitation_loss', policy_imitation_loss.cpu().item(), iteration_count)
    summary_writer.add_scalar('Training/Value_loss', value_loss.cpu().item(), iteration_count)
    summary_writer.add_scalar('Training/Expert_Iteration_loss', exit_loss.cpu().item(), iteration_count)
    summary_writer.add_scalar('Training/Total_loss', total_loss.cpu().item(), iteration_count)
    summary_writer.add_scalar('Training/Kullback-Leibler_divergence', kl_divergence.cpu().item(), iteration_count)

    summary_writer.add_scalar('Training/Mean_value_estimations', predictions['V'].mean().cpu().item(), iteration_count)
    summary_writer.add_scalar('Training/Mean_value_targets', values.mean().cpu().item(), iteration_count)
    summary_writer.add_scalar('Training/Apprentice_entropy', predictions['entropy'].mean().cpu().item(), iteration_count)
    summary_writer.add_scalar('Training/Expert_entropy', entropy(pi_mcts).mean().cpu().item(), iteration_count)


def log_opponent_modelling_loss_progress(summary_writer,
                                         opponent_modelling_loss,
                                         policy_inference_weight,
                                         kl_divergence_opponent_modelling,
                                         iteration_count):
    summary_writer.add_scalar('Training/Opponent_modelling_loss', opponent_modelling_loss.cpu().item(), iteration_count)
    summary_writer.add_scalar('Training/Policy_inference_weight', policy_inference_weight.cpu().item(), iteration_count)
    summary_writer.add_scalar('Training/Kullback-Leibler_divergence_opponent_modelling',
                              kl_divergence_opponent_modelling.cpu().item(), iteration_count)


def compute_opponent_modelling_loss(opponent_policy: torch.Tensor,
                                    opponent_s: torch.Tensor,
                                    apprentice_model: nn.Module) \
                                    -> Tuple:
    filtered_opponent_policies, filtered_opponent_s = filter_nan_tensors(
        opponent_policy,
        opponent_s)
    if len(filtered_opponent_policies) == 0:  # All sampled elements were nan
        opponent_modelling_loss = torch.Tensor([0.])
        policy_inference_weight = torch.Tensor([1.])
        kl_divergence = torch.Tensor([-1.])
    else:
        opponent_predictions = apprentice_model(filtered_opponent_s)

        opponent_modelling_loss = cross_entropy_loss(
            model_prediction=opponent_predictions['policy_0'],
            target=filtered_opponent_policies)

        # dynamically computed weight
        policy_inference_weight = 1 / (torch.sqrt(opponent_modelling_loss))

        # Assert that forall x_i in X: P(x_i) == 0 <-> Q(x_i). If
        # P(x_i) != 0 and Q(x_i) == 0.
        # Good explanation as to why:
        # https://stats.stackexchange.com/questions/97938/calculate-the-kullback-leibler-divergence-in-practice
        kl_divergence = torch.nn.functional.kl_div(
            opponent_predictions['policy_0'].log(),
            filtered_opponent_policies,  # To prevent log(0)
            reduction='batchmean'
        )
    return opponent_modelling_loss, policy_inference_weight, kl_divergence


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
