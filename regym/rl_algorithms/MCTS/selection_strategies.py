from typing import Dict
from math import sqrt, log, inf

from .sequential_node import SequentialNode


def old_UCB1(node, child, exploration_constant=sqrt(2)) -> float:
    '''
    More info: https://www.chessprogramming.org/UCT
    '''
    return (child.wins / child.visits +
            exploration_constant * sqrt(log(node.visits) / child.visits))


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
