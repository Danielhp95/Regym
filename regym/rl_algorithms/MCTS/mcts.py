from math import sqrt
import random
from .util import UCB1
from .open_loop_node import OpenLoopNode


def selection_phase(node, state, selection_policy=UCB1, selection_policy_args=[]):
    if not node.is_fully_expanded() or node.child_nodes == []:
        return node
    selected_node = sorted(node.child_nodes, key=lambda child: selection_policy(node, child, *selection_policy_args))[-1]
    state.step(selected_node.move)
    return selection_phase(selected_node, state)


def expansion_phase(node, state):
    if node.untried_moves != []:  # if we can expand (i.e. state/node is non-terminal)
        move = random.choice(node.untried_moves)
        state.step(move)
        node = node.add_child(move, state)
    return node


def rollout_phase(state):
    while (moves := state.get_moves()) != []:
        state.step(random.choice(moves))


def backpropagation_phase(node, state):
    if node is not None:
        node.update(state.get_result(node.player_just_moved))
        backpropagation_phase(node.parent_node, state)    


def action_selection_phase(node):
    return sorted(node.child_nodes, key=lambda c: c.wins / c.visits)[-1].move


def MCTS_UCT(rootstate, itermax, exploration_factor_ucb1=sqrt(2)):
    """ 
    Conducts a game tree search using the MCTS-UCT algorithm
    for a total of param itermax iterations. The search begins
    in the param rootstate. Assumes that 2 players are alternating
    with results being [0.0, 1.0].

    :param rootstate: The game state for which an action must be selected.
    :param itermax: number of MCTS iterations to be carried out. Also knwon as the computational budget.
    :returns: (int) Action that will be taken by an agent.
    """
    rootnode = OpenLoopNode(state=rootstate)
    
    for _ in range(itermax):
        node  = rootnode
        state = rootstate.clone()
        node  = selection_phase(node, state, selection_policy=UCB1, selection_policy_args=[exploration_factor_ucb1])
        node  = expansion_phase(node, state)
        rollout_phase(state)
        backpropagation_phase(node, state)
    
    return action_selection_phase(rootnode)
