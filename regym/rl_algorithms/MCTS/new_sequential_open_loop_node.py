from math import inf
from typing import Dict, List


class Node:

    def __init__(self, parent, player: int,
                 a: int, actions: List[int], priors: Dict[int, float]):
        self.N = 0  # Visits
        self.a = a  # Action taken from :param: parent to reach here
        self.player = player

        self.N_a = {a_i: 0 for a_i in actions}  # Child visitations
        self.W_a = {a_i: 0 for a_i in actions}  # Accumulated Child values
        self.Q_a = {a_i: 0 for a_i in actions}  # Mean Child values
        self.P_a = priors

        self.parent = parent
        self.children: Dict[int, object] = {}  # int -> Node

        self.actions = actions  # All possible actions that can be taken from this node
        self.untried_actions = actions[:]  # Actions leading to un-initiated nodes

        self.is_terminal = self.actions == []

    def add_child(self, a: int, P_a: Dict[int, float],
                  actions: List[int], player: int):
        self.children[a] = Node(parent=self, player=player, a=a, actions=actions, priors=P_a)
        self.untried_actions.remove(a)

    def is_fully_expanded(self):
        return self.untried_actions == []

    def update(self, a_i: int, value: float):
        self.W_a[a_i] += value
        # Equation to keep a running mean
        self.Q_a[a_i] = (self.N_a[a_i] * self.Q_a[a_i] + value) / (self.N_a[a_i] + 1)
        self.N_a[a_i] += 1
        # if abs(self.Q_a[a_i]) > 10000: import ipdb; ipdb.set_trace()

    def __repr__(self, indent=0, depth=inf):
        # This if statement is kind of uggly, right?
        if self.parent is not None:
            Q = self.parent.Q_a[self.a]
            W = self.parent.W_a[self.a]
            P = self.parent.P_a[self.a]
        elif len(self.children) > 0:  # For the root node
            Q = sum([self.Q_a[a_i] for a_i in self.children.keys()]) / len(self.children)
            W = sum([self.W_a[a_i] for a_i in self.children.keys()]) / len(self.children)
            P = 1.
        else: W, Q, P = 0., 0., 1.  # Root node before it has any children

        node_stats = '{}: Q={:.1f}. W={:.1f}/N={}. P={:.2f}.'.format(self.a, Q, W, self.N, P)
        terminal = '.Terminal' if self.is_terminal else ''
        prior = ''  # 'p={:.2f}'.format(self.prior) if self.prior else ''
        s = ('.' * indent) + node_stats + prior + terminal + '\n'
        if depth > 0:
            for a_i, child in self.children.items():
                s += child.__repr__(indent=indent + 1, depth=depth - 1)
        return s
