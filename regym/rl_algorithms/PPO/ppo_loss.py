from typing import Dict, List
import torch


def compute_loss(states: torch.Tensor, actions: torch.Tensor,
                 log_probs_old: torch.Tensor, returns: torch.Tensor,
                 advantages: torch.Tensor, model: torch.nn.Module,
                 ratio_clip: float, entropy_weight: float,
                 rnn_states: Dict[str, Dict[str, List[torch.Tensor]]] = None) -> torch.Tensor:
    '''
    Computes the loss of an actor critic model using the
    loss function from equation (9) in the paper:
    Proximal Policy Optimization Algorithms: https://arxiv.org/abs/1707.06347

    :param states: Dimension: batch_size x state_size: States visited by the agent.
    :param actions: Dimension: batch_size x action_size. Actions which the agent
                    took at every state in :param states: with the same index.
    :param log_probs_old: Dimension: batch_size x 1. Log probability of taking
                          the action with the same index in :param actions:.
                          Used to compute the policy probability ratio.
                          Refer to original paper equation (6)
    :param returns: Dimension: batch_size x 1. Empirical returns obtained via
                    calculating the discounted return from the environment's rewards
    :param advantages: Dimension: batch_size x 1. Estimated advantage function
                       for every state and action in :param states: and
                       :param actions: (respectively) with the same index.
    :param model: torch.nn.Module used to compute the policy probability ratio
                  as specified in equation (6) of original paper.
    :param ratio_clip: Epsilon value used to clip the policy ratio's value.
                       This parameter acts as the radius of the Trust Region.
                       Refer to original paper equation (7).
    :param entropy_weight: Coefficient to be used for the entropy bonus
                           for the loss function. Refer to original paper eq (9)
    :param rnn_states: The :param model: can be made up of different submodules.
                       Some of these submodules will feature an LSTM architecture.
                       This parameter is a dictionary which maps recurrent submodule names
                       to a dictionary which contains 2 lists of tensors, each list
                       corresponding to the 'hidden' and 'cell' states of
                       the LSTM submodules. These tensors are used by the
                       :param model: when calculating the policy probability ratio.
    '''
    if rnn_states is not None:
        prediction = model(states, actions, rnn_states=rnn_states)
    else:
        prediction = model(states, actions)

    ratio = (prediction['log_pi_a'] - log_probs_old).exp()
    obj = ratio * advantages
    obj_clipped = ratio.clamp(1.0 - ratio_clip,
                              1.0 + ratio_clip) * advantages
    policy_loss = -1. * torch.min(obj, obj_clipped).mean() - entropy_weight * prediction['ent'].mean() # L^{clip} and L^{S} from original paper
    value_loss = 0.5 * torch.nn.functional.mse_loss(returns, prediction['v'])
    total_loss = (policy_loss + value_loss)
    return total_loss
