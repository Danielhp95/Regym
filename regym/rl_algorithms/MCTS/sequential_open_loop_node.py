from functools import reduce
import gym


class SequentialOpenLoopNode:
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

    Note: self.wins is from the perspective of player_just_moved.

    TODO: write assumptions made over the interface of the underlying
          OpenAIGym environment. THIS IS SUPER IMPORTANT
    """
    def __init__(self, move: int = None, parent=None, prior: float = None,
                 state: gym.Env = None):
        self.move = move  # Move that was taken to reach this game state
        self.parent_node = parent  # "None" for the root node
        self.child_nodes = []

        self.prior = prior  # Used only for PUCT calculations
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_moves()  # Future childNodes
        self.player_just_moved = state.player_just_moved  # To check who won or who lost.

    def is_fully_expanded(self):
        return self.untried_moves == []

    def add_child(self, move: int, state: gym.Env, prior: float = None):
        """
        Adds a new child node to this Node.
        :param move: (int) action taken by the player
        :param state: (GameState) state corresponding to new child node
        :returns: new expanded node added to the tree
        """
        node = SequentialOpenLoopNode(move=move, parent=self, state=state, prior=prior)

        self.untried_moves.remove(move)
        self.child_nodes.append(node)
        return node

    def update(self, result):
        """
        Updates the node statistics saved in this node with the param result
         which is the information obtained during the latest rollout.
        :param result: (bool) 1 for victory, 0 for draw / loss.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self, indent=0) -> str:
        '''
        For debugging purposes. Prints the tree starting with
        :param: self as the root with corresponding statistics.
        Source code is messy, it was made in a jam ;).
        '''
        prior = 'p={:.2f}'.format(self.prior) if self.prior else ''
        s = reduce(lambda acc, x: x + acc, ['.' for i in range(indent)], '') \
            + f'{self.move}: {self.wins}/{self.visits} ' + prior + '\n'
        for n in self.child_nodes:
            s += n.__repr__(indent=indent + 1)
        return s
