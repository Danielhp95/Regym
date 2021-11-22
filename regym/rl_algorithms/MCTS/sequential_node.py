from math import inf
from typing import Dict, List


class SequentialNode:

    def __init__(self, parent: 'SequentialNode', player: int,
                 a: int, actions: List[int], priors: Dict[int, float]):
        self.N: int = 0  # Visits
        self.a: int = a  # Action taken from :param: parent to reach here
        self.player: int = player  # Player who would take an action in the state represented in this node

        self.N_a: Dict[int, int] = {a_i: 0 for a_i in actions}  # Child visitations
        self.W_a: Dict[int, float] = {a_i: 0. for a_i in actions}  # Accumulated Child values
        self.Q_a: Dict[int, float] = {a_i: 0. for a_i in actions}  # Mean Child values
        self.P_a: Dict[int, float] = priors

        self.parent: 'SequentialNode' = parent
        self.children: Dict[int, 'SequentialNode'] = {}

        self.actions: List[int] = actions  # All possible actions that can be taken from this node
        self.untried_actions: List[int] = actions[:]  # Actions leading to un-initiated nodes

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        # If a node has no expanded children, it's at the fringe of the tree.
        return self.children == {}

    @property
    def is_terminal(self) -> bool:
        # If no actions can be taken, it means that there are no further
        # transitions to explore, we are in a terminal node.
        return self.actions == []

    def add_child(self, a: int, P_a: Dict[int, float],
                  actions: List[int], player: int):
        self.children[a] = SequentialNode(parent=self, player=player, a=a,
                                          actions=actions, priors=P_a)
        self.untried_actions.remove(a)

    def is_fully_expanded(self) -> bool:
        return self.untried_actions == []

    def update_edge_statistics(self, a_i: int, value: float):
        self.W_a[a_i] += value
        # Equation to keep a running mean
        self.Q_a[a_i] = (self.N_a[a_i] * self.Q_a[a_i] + value) / (self.N_a[a_i] + 1)
        self.N_a[a_i] += 1

    def __repr__(self, indent=0, depth=inf) -> str:
        '''
        Prints current node to the terminal, following the current
        node up to a maximum of :param: depth traversals. Useful for debugging.

        :param indent: Number of white space indentations.
        :param depth: Maximum depth to print children.
        :returns: String representation of this node
        '''
        # This if statement is kind of ugly, right?
        if not self.is_root:
            Q = self.parent.Q_a[self.a]
            W = self.parent.W_a[self.a]
            P = self.parent.P_a[self.a]
        elif len(self.children) > 0:  # For the root node
            Q = sum([self.Q_a[a_i] for a_i in self.children.keys()]) / len(self.children)
            W = sum([self.W_a[a_i] for a_i in self.children.keys()]) / len(self.children)
            P = 1.
        else: W, Q, P = 0., 0., 1.  # Root node before it has any children

        node_stats = '{}: P: {}. Q={:.1f}. W={:.1f}/N={}. Pr={:.2f}.'.format(self.a, self.player, Q, W, self.N, P)
        terminal = '.Terminal' if self.is_terminal else ''
        s = ('.' * indent) + node_stats + terminal + '\n'
        if depth > 0:
            for a_i, child in self.children.items():
                s += child.__repr__(indent=indent + 1, depth=depth - 1)
        return s
