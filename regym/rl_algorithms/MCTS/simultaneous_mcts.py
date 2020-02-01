from math import sqrt
import random
from .util import UCB1
from .simultaneous_open_loop_node import SimultaneousOpenLoopNode


def selection_phase(nodes, state, selection_policy=UCB1, selection_policy_args=[]):
    expanded = [False for n in nodes]
    while not all(expanded):
        moves, expanded = choose_moves(nodes, selection_policy, selection_policy_args)
        state.step(moves)
        nodes = [n.doStuff(moves, state) for n in nodes]
    return nodes


def choose_moves(self, nodes, selection_policy, selection_policy_args):
    expanded = []
    moves = []
    for n in nodes:
        if n.is_fully_expanded():
            expanded.append(False)
            selected_node = sorted(n.child_nodes,
                                   key=lambda child: selection_policy(n, child, *selection_policy_args))[-1]
            moves.append(selected_node.move)
        else:
            expanded.append(True)
            moves.append(random.choice(n.untried_moves))
    return moves, expanded


def rollout_phase(state):
    # TODO: Currently we just stop once we reach the end of the tree
    # TODO: Currently get_result() returns True/False...this needs to be a numeric score for Back Prop
    return state


def backpropagation_phase(nodes, state):
    for n in nodes:
        if n is not None:
            #TODO: get_result() returns True/False...we need a numeric score (evaluation) for
            # each player in the state
            n.update(state.get_result(n.perspective_player))
            backpropagation_phase(n.parent_node, state)


def action_selection_phase(nodes):
    return [sorted(n.child_nodes, key=lambda c: c.wins / c.visits)[-1].move
            for n in nodes]


def MCTS_UCT(rootstate, itermax, exploration_factor_ucb1=sqrt(2), player_count = 2):
    """ 
    Conducts a game tree search using the MCTS-UCT algorithm
    for a total of param itermax iterations. The search begins
    in the param rootstate. Defaults to 2 players with results in [0.0, 1.0].

    :param rootstate: The game state for which an action must be selected.
    :param itermax: number of MCTS iterations to be carried out. Also knwon as the computational budget.
    :returns: List[int] Action that will be taken by EACH player.
    """
    root_nodes = [SimultaneousOpenLoopNode(state=rootstate, perspective_player=i)
                  for i in range(player_count)]
    for _ in range(itermax):
        nodes = root_nodes
        state = rootstate.clone()
        nodes = selection_phase(nodes, state, selection_policy=UCB1, selection_policy_args=[exploration_factor_ucb1])
        rollout_phase(state)
        backpropagation_phase(nodes, state)

    return action_selection_phase(root_nodes)
