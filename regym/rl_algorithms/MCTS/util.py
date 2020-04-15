from math import sqrt, log


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
