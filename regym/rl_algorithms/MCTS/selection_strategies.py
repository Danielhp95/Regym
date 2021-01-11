from typing import Dict
from math import sqrt, log, inf

from .sequential_node import SequentialNode


def old_UCB1(node, child, exploration_constant=sqrt(2)) -> float:
    '''
    More info: https://www.chessprogramming.org/UCT
    '''
    return (child.wins / child.visits +
            exploration_constant * sqrt(log(node.visits) / child.visits))


def old_PUCT(node, child, c: float) -> float:
    '''
    (P)redictor UCB1. A modification of UCB1 used to predict the value of
    child node :param: child.

    :param node: Node whose :param: child is evaluated
    :param child: Child node of :param: nodejfor which the PUCT
                  score is computed
    :param c: Exploration constant
    '''
    Q = child.wins / child.visits if child.visits > 0 else 0
    U = c * child.prior * sqrt(node.visits) / (child.visits + 1)
    return Q + U


def UCB1(node: SequentialNode, c: float) -> Dict[int, float]:
    '''
    :param node: Node whose children are evaluated
    :param c: Exploration constant
    :returns: Dictionary containing the selection score for each
              child node present in :param: node
    '''
    return {a_i: inf if node.N_a[a_i] == 0 else
                 node.Q_a[a_i] + c * sqrt(log(node.N) / (node.N_a[a_i]))
            for a_i in node.actions}


def PUCT(node: SequentialNode, c: float) -> Dict[int, float]:
    '''
    (P)redictor UCB1. A modification of UCB1 used to predict the most promising
    child node for :param: node.

    :param node: Node whose children are evaluated
    :param c: Exploration constant
    :returns: Dictionary containing the selection score for each
              child node present in :param: node
    '''
    return {a_i: node.Q_a[a_i] +
            c * node.P_a[a_i] * (sqrt(node.N) / (node.N_a[a_i] + 1))
            for a_i in node.actions}
