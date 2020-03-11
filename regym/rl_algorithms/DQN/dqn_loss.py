from typing import Dict, List

import torch
from torch.utils.tensorboard import SummaryWriter

summary_writer = None

def compute_loss(states: torch.Tensor,
                 actions: torch.Tensor,
                 next_states: torch.Tensor,
                 rewards: torch.Tensor,
                 non_terminals: torch.Tensor,
                 model: torch.nn.Module,
                 target_model: torch.nn.Module,
                 gamma: float = 0.99,
                 use_PER: bool = False,
                 #PER_beta: float = 1.0,
                 importanceSamplingWeights: torch.Tensor = None,
                 #use_HER: bool = False,
                 #summary_writer: object = None,
                 iteration_count: int = 0,
                 rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None) -> torch.Tensor:
    '''
    :param states: Dimension: batch_size x state_size: States visited by the agent.
    :param actions: Dimension: batch_size x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
    :param next_states: Dimension: batch_size x state_size: Next states visited by the agent.
    :param non_terminals: Dimension: batch_size x 1: Non-terminal integers.
    :param rewards: Dimension: batch_size x 1. Environment rewards.
    :param model: torch.nn.Module used to compute the loss.
    :param target_model: torch.nn.Module used to compute the loss.
    :param gamma: float discount factor.
    :param rnn_states: The :param model: can be made up of different submodules.
                       Some of these submodules will feature an LSTM architecture.
                       This parameter is a dictionary which maps recurrent submodule names
                       to a dictionary which contains 2 lists of tensors, each list
                       corresponding to the 'hidden' and 'cell' states of
                       the LSTM submodules. These tensors are used by the
                       :param model: when calculating the policy probability ratio.
    '''
#    td_error = compute_td_error(states, actions, next_states,
#                                rewards, non_terminals, gamma,
#                                model, target_model)
#
    prediction = model(states)
    target_prediction = target_model(states, action=prediction['action'])

    q_values = prediction['q_values']
    q_values_from_selected_actions = q_values.gather(dim=1, index=actions.long().unsqueeze(1))

    target_q_values_succ_state = target_prediction['q_values'].detach()
    max_target_q_values_succ_state, _ = target_q_values_succ_state.max(1)

    max_target_q_values_succ_state = max_target_q_values_succ_state.view(-1, 1)

    # Compute the expected Q values
    td_target = rewards + non_terminals * (gamma * max_target_q_values_succ_state)

    # Compute td_error:
    td_error = td_target.detach() - q_values_from_selected_actions

    if use_PER:  # TODO: worry about this later
        diff_squared = importance_sampling_weights.unsqueeze(1) * td_error.pow(2.0)
    else:
        diff_squared = td_error.pow(2.0)

    loss = 0.5 * torch.mean(diff_squared)

    if summary_writer is not None:
        paper_q_value = prediction['q_values'].gather(dim=1, index=torch.ones(prediction['q_values'].size()[0], 1).long())
        summary_writer.add_scalar('Training/Mean_paper_q_value', paper_q_value.cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/Mean_q_values', prediction['q_values'].cpu().mean().item(), iteration_count)
        summary_writer.add_scalar('Training/Std_q_values', prediction['q_values'].cpu().std().item(), iteration_count)
        summary_writer.add_scalar('Training/Q_value_loss', loss.cpu().item(), iteration_count)
        summary_writer.add_scalar('Training/Q_value_entropy', prediction['entropy'].mean().cpu().item(), iteration_count)
    return loss


def compute_td_error(states: torch.Tensor,
                     actions: torch.Tensor,
                     next_states: torch.Tensor,
                     rewards: torch.Tensor,
                     non_terminals: torch.Tensor,
                     gamma: float,
                     model: torch.nn.Module,
                     target_model: torch.nn.Module) -> torch.Tensor:
    '''
    TODO: document, further break down into functions
    '''

    prediction = model(states)
    target_prediction = target_model(states, action=prediction['action'])

    q_values = prediction['q_values']
    q_values_from_selected_actions = q_values.gather(dim=1, index=actions.long().unsqueeze(1))

    target_q_values_succ_state = target_prediction['q_values'].detach()
    max_target_q_values_succ_state, _ = target_q_values_succ_state.max(1)

    max_target_q_values_succ_state = max_target_q_values_succ_state.view(-1, 1)

    # Compute the expected Q values
    td_target = rewards + non_terminals * (gamma * max_target_q_values_succ_state)

    # Compute td_error:
    td_error = td_target.detach() - q_values_from_selected_actions
    return td_error
