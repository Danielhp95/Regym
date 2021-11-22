from math import inf
from typing import List, Dict

import numpy as np


def extract_best_actions(scores: Dict[int, float]) -> List[int]:
    '''
    Extracts the (potentially multiple) highest scoring actions (keys)
    from :param: scores, containing a maping from actions to scores.
    :param scores: rankings of each action according to a selection strategy
    :returns: List of most valuable actions.
    '''
    best_candidates, best_value = [], -inf
    for a_i, score in scores.items():
        if score < best_value: continue
        if score == best_value: best_candidates.append(a_i)
        if score > best_value: best_candidates = [a_i]; best_value = score
    return best_candidates


def add_dirichlet_noise(alpha: float,
                        p: Dict[int, float],
                        noise_strength: float=1.) -> Dict[int, float]:
    '''
    Adds dirichlet noise to distribution stored in the values of :param p:.
    :param alpha: Parameter of Dirichlet distribution
    :param p: Dictionary whose values contain the distribution to be perturbed
    :param noise_strength: Ranging [0, 1], Multiplier for noise strenght
    :returns: Dictionary containing perturbed distribution
    '''
    assert 0 <= noise_strength <= 1, ('Param noise_strength must lie between 0 and 1. '
                                        f'Given: {noise_strength}.')

    dirichlet_noise = np.random.dirichlet(np.full(shape=(len(p.keys())), fill_value=alpha))
    scaled_dirichlet_noise = noise_strength * dirichlet_noise
    dirchlet_noise_total = sum(scaled_dirichlet_noise)

    p_total = sum(p.values())
    perturbed_priors_total = p_total + dirchlet_noise_total
    return {a_i: (p_a_i + scaled_dirichlet_noise[i]) / perturbed_priors_total
            for i, (a_i, p_a_i) in enumerate(p.items())}


def random_selection_policy(obs,
                            legal_actions: List[int],
                            self_player_index: int = None,
                            requested_player_index: int = None,
                            action_dim: int = -1) -> np.ndarray:
    '''
    TODO:
    '''
    if legal_actions == []: return []
    num_legal_actions = len(legal_actions)
    action_probability = 1 / num_legal_actions
    return np.array(
        [action_probability if a_i in legal_actions else 0.
         for a_i in range(action_dim)]
    )
