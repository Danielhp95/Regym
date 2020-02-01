class SimultaneousOpenLoopNode():
    """
        An Open-loop approach keeps only one state representation at the root node of the game tree being computed by MCTS, and each of the edges represent an action. As the game tree is being traversed, the actions stored in the nodes being traversed are used as input to the model of the game. Thus the initial state stored at the root of the tree changes according to the actions selected during the game tree traversal. Once an MCTS iteration is over, this stated modified during the tree traversal is discarded, and a new clone of the state of the root node is created. Furthermore, we don't store the intermediate states obtained during the traversal. This means that we are esentially throwing away the computation performed by the model as soon as we don't need it.


        Node of a game tree. A tree is a connected acyclic graph.
        Note: self.wins is from the perspective of playerJustMoved.

        TODO: write assumptions made over the interface of the underlying
              OpenAIGym environment. THIS IS SUPER IMPORTANT
    """
    def __init__(self, perspective_player: int, move=None, parent=None, state=None):
        self.move = move  # Move that was taken to reach this game state
        self.parent_node = parent  # "None" for the root node
        self.child_nodes = []
        self.perspective_player = perspective_player

        self.wins = 0
        self.visits = 0
        self.is_chance_node = state is None
        self.untried_moves = state.get_moves(self.perspective_player) if not self.is_chance_node else None
   #     self.player_just_moved = state.player_just_moved  # To check who won or who lost.

    def is_fully_expanded(self):
        assert not self.is_chance_node
        return self.untried_moves == []

    def doStuff(self, moves, whereWeEndUpAfterwards):
        # we iterate through actions and move to the next node
        # if none exist then we expand one and continue
        node = self
        for (i, m) in enumerate(moves):
            matching_nodes = list(filter(lambda n: n.move == m, node.child_nodes))
            assert len(matching_nodes) == 0 or len(matching_nodes) == 1
            if len(matching_nodes) == 0:
                # Node must be expanded
                # For all other players we add a chance node, and we end on a decision node (state required)
                node = node.add_child(m, None if i == (len(moves)-1) else whereWeEndUpAfterwards)
            else:
                node = matching_nodes.pop(0)
        # Then return final node after all players have acted
        return node


    def add_child(self, move, state=None):
        """
        Adds a new child node to this Node.
        :param move: (int) action taken by the player
        :param state: (GameState) state corresponding to new child node
        :returns: new expanded node added to the tree
        """
        node = SimultaneousOpenLoopNode(self.perspective_player, move=move, parent=self, state=state)
        if not self.is_chance_node: self.untried_moves.remove(move)
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
