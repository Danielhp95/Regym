# TODO: give all credit to Juyenesh
import math
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
from scipy.special import softmax


def compute_nash_averaging(payoff_matrix: np.ndarray, perform_logodds_transformation=False) \
                           -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes maximum entropy Nash equilibrium and Nash Averaging
    for :param payoff_matrix:.

    For more details see:
        Balduzzi et al., "Re-evaluating Evaluation", 2018,
        https://arxiv.org/abs/1806.02643

    :param payoff_matrix: For representing payoffs for player 1 of a symmetric 2-player game.
    :param perform_logodds_transformation: (default: False). TODO: explain why
    :returns: Maximum entropy Nash Equilibrium vector, Nash Averaging vector
    '''
    game_matrix = preprocess_matrix(payoff_matrix, perform_logodds_transformation)
    check_validity(game_matrix, perform_logodds_transformation)
    maxent_nash, nash_averaging = compute_nash_average(game_matrix, steps=2**10)
    return maxent_nash, nash_averaging


def compute_nash_average(payoff_matrix: np.ndarray, **method_kwargs):
    """ Computes the maxent Nash/Correlated Equilibrium and the associated nash average ranking
        Arguments
            - `payoff_matrix`:
            - `method`:
            - `**method_kwargs`
        Returns
            - `NERes`:
    For more details see:
        Balduzzi et al., "Re-evaluating Evaluation", 2018,
        https://arxiv.org/abs/1806.02643
    """
    strategy = compute_maxent_correlated_equilibrium(payoff_matrix, **method_kwargs)
    nash_avg = (payoff_matrix @ strategy.reshape(-1, 1)).ravel()
    return (strategy, nash_avg)


def compute_maxent_correlated_equilibrium(payoff_matrix: np.ndarray, steps: int,
                                          eps: Optional[float] = None,
                                          tol: float = 1e-8) \
                                          -> np.ndarray:
    antisym_payoff = np.stack([payoff_matrix, -payoff_matrix])
    sol = solve_maxent_ce(antisym_payoff, tol=tol, eps=eps, steps=steps)
    assert (np.abs(sol - sol.T) < tol).all(), np.abs(sol - sol.T).max() # TODO check what this is, maybe create assertio that throws a NumericalError or something like this
    strategy = sol.sum(axis=1)  # sum along the columns for player 1 strategy
    return strategy


def solve_maxent_ce(payoffs: np.ndarray, steps: int, eps: Optional[float] = None,
                    tol: float = 1e-8) -> np.ndarray:
    """Solves for the MaxEntropy Correlated Equilibrium given the payoff matrices

    Parameters
    ----------
    payoffs : np.ndarray
        payoff matrices of all players
    steps : int
        Max number of steps of gradient descent
    eps : Optional[float], optional
        Small constant for avoiding divide-by-zero, by default None
    tol : float, optional
        Tolerance for CE computation, by default 1e-8

    Returns
    -------
    np.ndarray
        MaxEnt CE policy for the players
    For more details see:
        Ortiz et al., "Maximum entropy correlated equilibria", 2007,
        http://proceedings.mlr.press/v2/ortiz07a/ortiz07a.pdf
    """
    if eps is None:
        eps = np.finfo(np.float).eps

    N = payoffs.shape[0]
    action_counts = payoffs.shape[1:]

    c = sum(np.abs(payoff_gain(payoffs[i].swapaxes(0, i))).sum(axis=0).max() for i in range(N))
    c = max(c, 1) # Just in case that c is 0

    lr = 0.9 / c

    lambdas = [lr * np.ones((i, i)) for i in action_counts]
    for i in range(N):
        rac = np.arange(action_counts[i])
        lambdas[i][rac, rac] = 0.0

    prev_policy = None
    for _ in range(steps):
        log_policy = get_log_gibbs_pi(payoffs, lambdas)
        policy = np.exp(log_policy)
        if prev_policy is not None:
            if np.abs(policy - prev_policy).max() < tol:
                break

        pos_regret = get_regret(policy, payoffs, positive=True)
        neg_regret = get_regret(policy, payoffs, positive=False)

        for i in range(N):
            ac = action_counts[i]
            rac = np.arange(ac)
            # Eqn 4
            chg = ((pos_regret[i] + eps) / (pos_regret[i] + neg_regret[i] + 2 * eps)) - 0.5
            chg[rac, rac] = 0.0
            delta = lr * chg
            # Eqn 2
            lambdas[i] += delta
            np.clip(lambdas[i], 0.0, None, lambdas[i])

        prev_policy = policy
    return policy


def payoff_gain(payoff: np.ndarray) -> np.ndarray:
    """ Returns the player's payoff gain matrix from playing action other than

        Arguments:
            - `payoff`:
        Returns:
            - `gain_mat`:
    """
    assert payoff.ndim == 2
    gain_mat = payoff[:, np.newaxis, :] - payoff[np.newaxis, :, :]
    return gain_mat


def get_log_gibbs_pi(payoffs: np.ndarray, lambdas: List[np.ndarray]) -> np.ndarray:
    """

        For more details see:
            Ortiz et al., "Maximum entropy correlated equilibria", 2007,
            http://proceedings.mlr.press/v2/ortiz07a/ortiz07a.pdf
    """
    #
    # Theorem 1
    N = payoffs.shape[0]
    log_policy = np.zeros(payoffs.shape[1:])

    for i in range(N):
        lam = lambdas[i]
        payoff = payoffs[i]
        payoff_perm = payoff.swapaxes(0, i)
        perm_shape = payoff_perm.shape
        gain_mat = payoff_gain(payoff_perm)
        # sum_{a'_i != a_i} lambda_{i, a_i, a'_i} * G_i(a'_i, a_i, a_{-i})
        tmp = ((lam.swapaxes(0, 1)[:, :, np.newaxis] * gain_mat).sum(axis=0).reshape(perm_shape)
               .swapaxes(0, i))
        log_policy -= tmp

    log_pi = np.log(softmax(log_policy))
    return log_pi


def get_regret(policy: np.ndarray, payoffs: np.ndarray, positive: bool = True) -> List[np.ndarray]:
    N = payoffs.shape[0]
    action_counts = payoffs.shape[1:]

    ret = []
    for i in range(N):
        ac = action_counts[i]
        payoff = payoffs[i]

        policy_perm = policy.swapaxes(0, i)
        payoff_perm = payoff.swapaxes(0, i)

        gain_mat = payoff_gain(payoff_perm)
        r_mat = gain_mat.swapaxes(0, 1) * policy_perm[:, np.newaxis, :]
        if not positive:
            r_mat = -r_mat

        np.clip(r_mat, 0.0, None, r_mat)
        ret.append(r_mat.reshape(ac, ac, -1).sum(axis=2))

    return ret


def preprocess_matrix(payoff_matrix, perform_logodds_transformation):
    game_matrix = payoff_matrix
    if not isinstance(game_matrix, np.ndarray): game_matrix = np.array(game_matrix)
    if perform_logodds_transformation:
        epsilon = 1e-10
        # Modyfing values near 0 and 1 to prevent
        # infinities after log-odds operation
        game_matrix = np.where(np.isclose(game_matrix, 0),
                               game_matrix + epsilon, game_matrix)
        game_matrix = np.where(np.isclose(game_matrix, 1),
                               game_matrix - epsilon, game_matrix)
        game_matrix = np.log(game_matrix/(1 - game_matrix))
    return game_matrix


def is_matrix_antisymmetrical(m: np.array) -> bool:
    '''
    TODO: This is already in Regym, eliminate once we merge
    '''
    return m.shape[0] == m.shape[1] and np.allclose(m, -1 * m.T, rtol=1e-03, atol=1e-03)


def check_validity(payoff_matrix: np.ndarray, perform_logodds_transformation: bool):
    is_matrix_square = lambda m: m.ndim == 2 and m.shape[0] == m.shape[1]
    if payoff_matrix.dtype.kind not in np.typecodes['AllInteger'] and \
       payoff_matrix.dtype.kind not in np.typecodes['AllFloat']: raise ValueError('Input payoff_matrix should contain floats or integers')
    if not is_matrix_square(payoff_matrix):
        raise ValueError('Payoff matrix should be a 2D and square.')
    if not perform_logodds_transformation and not is_matrix_antisymmetrical(payoff_matrix):
        raise ValueError('''
                         Input payoff_matrix was not antisymmetrical. Nash averaging can only
                         safely be computed on an antisymmetrical matrix because otherwise
                         it is not guaranteed that there is a unique maximum entropy Nash
                         equilibria, and thus we lose the property of "interpretability"
                         from Nash Averaging.
                         
                         If the payoff_matrix is an empirical winrate matrix (TODO, define further),
                         then set :param perform_logodds_transformation: to True to turn the
                         payoff_matrix into an antisymmetrical matrix as defined in 
                         Balduzzi 2018 Revaluating Evaluation.
                         ''')
