from math import sqrt, log

def UCB1(node, child, exploration_constant=sqrt(2)):
    return child.wins / child.visits + exploration_constant * sqrt(log(node.visits) / child.visits)
