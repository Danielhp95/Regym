import torch


# rnn_states, dictionary of nn modules. values features two lists. 1) List of hidden states 2) List of cell states.
def compute_loss(states, actions, log_probs_old, returns, advantages, model, ratio_clip, entropy_weight, rnn_states=None):
    '''
    TODO: document
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
    value_loss = 0.5 * (returns - prediction['v']).pow(2).mean()
    total_loss = (policy_loss + value_loss)
    return total_loss
