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


def add_dirichlet_noise(alpha: float, p: Dict[int, float]) -> Dict[int, float]:
    '''
    Adds dirichlet noise to distribution stored in the values of :param p:.
    :param alpha: Parameter of Dirichlet distribution
    :param p: Dictionary whose values contain the distribution to be perturbed
    :returns: Dictionary containing perturbed distribution
    '''
    dirichlet_noise = np.random.dirichlet(np.full(shape=(len(p.keys())), fill_value=alpha))
    dirchlet_noise_total = sum(dirichlet_noise)
    p_total = sum(p.values())
    perturbed_priors_total = p_total + dirchlet_noise_total
    return {a_i: (p_a_i + dirichlet_noise[i]) / perturbed_priors_total
            for i, (a_i, p_a_i) in enumerate(p.items())}
