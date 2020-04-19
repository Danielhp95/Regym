from typing import Dict
from math import sqrt, log, inf


def UCB1(node, child, exploration_constant=sqrt(2)):
    '''
    More info: https://www.chessprogramming.org/UCT
    '''
    return child.wins / child.visits + exploration_constant * sqrt(log(node.visits) / child.visits)


def PUCT(node, child, c: float):
    '''
    (P)redictor UCB1. A modification of UCB1 which a predicion of a good arm.
    :param node: TODO
    :param child: Child for which the PUCT value is being calculated
    :param c: Exploration constant
    '''
    Q = child.wins / child.visits if child.visits > 0 else 0
    U = c * child.prior * sqrt(node.visits) / (child.visits + 1)
    return Q + U


def new_UCB1(node, c: float) -> Dict[int, float]:
    return {a_i: inf if node.N_a[a_i] == 0 else \
                node.Q_a[a_i] + c * sqrt(log(node.N) / (node.N_a[a_i]))
            for a_i in node.actions}


def new_PUCT(node, c: float) -> Dict[int, float]:
    return {a_i: node.Q_a[a_i] + \
                 c * node.P_a[a_i] * (sqrt(node.N) / (node.N_a[a_i] + 1))
            for a_i in node.actions}
