from typing import List
from functools import reduce
import gym


class SimultaneousOpenLoopNode:
    """
    Open loop tree implementation apt for MCTS in SIMULTANEOUS tasks.
    An Open-loop approach keeps only one state representation at the root node
    of the game tree being computed by MCTS. Except the root node, each node
    stores which action was executed w.r.t to its parent node. As the game
    tree is being traversed, the actions stored in the nodes being traversed
    are used as input to the model of the game. Thus the initial state stored
    at the root of the tree changes according to the actions selected during
    the game tree traversal.
    Once an MCTS iteration is over, this stated modified during the tree
    traversal is discarded, and a new clone of the state of the root node
    is created.


    TODO: write assumptions made over the interface of the underlying
          OpenAIGym environment. THIS IS SUPER IMPORTANT
    """

    def __init__(self, perspective_player: int, move: int = None,
                 parent=None, state: gym.Env = None):
        self.move = move           # Move that was taken to reach this game state
        self.parent_node = parent  # "None" for the root node
        self.child_nodes = []
        self.perspective_player = perspective_player

        self.wins = 0  # Note: self.wins is from the perspective of `perspective_player`.

        self.visits = 0
        self.is_chance_node = state is None  # i.e `perspective_player` does not act in this node
        self.untried_moves = state.get_moves(self.perspective_player) if not self.is_chance_node else None

    def is_fully_expanded(self) -> bool:
        assert not self.is_chance_node
        return self.untried_moves == []

    def descend_and_expand(self, moves: List[int], state: gym.Env):
        '''
        Descends the tree, of which `self` is a node, according
        to :param: moves. If there are no deeper nodes containing
        the moves from :param: moves, then the tree is expanded
        to contain them.

        :params state: Game state
        :params moves: List of moves coming from selection phase,
                       one for each player
        :returns: Child node, linked backwards to `self` by :param: moves
        '''
        # we iterate through actions and move to the next node
        # if none exist then we expand one and continue
        def descend_tree(node, move_taken: int, chance_node=True):
            matching_node = list(filter(lambda n: n.move == move_taken, node.child_nodes))
            # Node must be expanded
            if matching_node == []: return node.add_child(
                                                      move_taken,
                                                      state if not chance_node else None)
            # Node exists
            else: return matching_node.pop(0)

        # The first action that must be taken is that from `self.perspective_player`
        node = descend_tree(self, moves[self.perspective_player])
        last_index = len(moves) - 1
        last_node_index = last_index - 1 if self.perspective_player == last_index else last_index
        for (i, m) in enumerate(moves):
            if i == self.perspective_player:
                continue
            node = descend_tree(node, m, chance_node=(last_node_index != i))
        return node  # Final node, after all players have acted

    def add_child(self, move: int, state: gym.Env = None):
        """
        Adds a new child node to this Node.
        :param move: action taken by the player
        :param state: state corresponding to new child node
        :returns: new expanded node added to the tree
        """
        node = SimultaneousOpenLoopNode(self.perspective_player, move=move, parent=self, state=state)
        if not self.is_chance_node: self.untried_moves.remove(move)
        self.child_nodes.append(node)
        return node

    def update(self, result: float):
        """
        Updates the node statistics saved in this node with the param result
         which is the information obtained during the latest rollout.
        :param result: (bool) 1 for victory, 0 for draw / loss.
        """
        self.visits += 1
        self.wins += result

    def depth(self):
        '''
        Computes the maximum depth of the tree starting at node :param: self
        '''
        if self.child_nodes == []: return 0
        else: return 1 + max([c.depth() for c in self.child_nodes])

    def __repr__(self, indent=0, ignore_chance=True) -> str:
        '''
        For debugging purposes. Prints the tree starting with
        :param: self as the root with corresponding statistics.
        Source code is messy, it was made in a jam ;).
        '''
        s = ''
        if self.parent_node is not None and self.parent_node.is_chance_node and ignore_chance:
            pass
        else:
            s += reduce(lambda acc, x: x + acc, ['.' for i in range(indent)], '') \
                 + f'{self.move}: {self.wins}/{self.visits}\n'
        for n in self.child_nodes:
            s += n.__repr__(indent=indent + 1)
        return s
