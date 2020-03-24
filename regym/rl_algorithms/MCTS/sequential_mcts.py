from math import sqrt
import random
from .util import UCB1
from .sequential_open_loop_node import SequentialOpenLoopNode


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


def rollout_phase(state, rollout_budget: int):
    for i in range(rollout_budget):
        moves = state.get_moves()
        if moves == []: return
        state.step(random.choice(state.get_moves()))


def backpropagation_phase(node, state):
    if node is not None:
        node.update(state.get_result(node.player_just_moved))
        backpropagation_phase(node.parent_node, state)


def action_selection_phase(node):
    return sorted(node.child_nodes, key=lambda c: c.wins / c.visits)[-1].move


def MCTS_UCT(rootstate, budget: int, num_agents: int,
             rollout_budget = 100000,
             exploration_factor_ucb1: float = sqrt(2)):
    """
    Conducts a game tree search using the MCTS-UCT algorithm
    for a total of param itermax iterations. The search begins
    in the param rootstate. Assumes that 2 players are alternating
    with results being [0.0, 1.0].

    :param rootstate: The game state for which an action must be selected.
    :param budget: number of MCTS iterations to be carried out. Also knwon as the computational budget.
    :param num_agents: UNUSED
    :returns: (int) Action that will be taken by an agent.
    """
    rootnode = SequentialOpenLoopNode(state=rootstate)

    for _ in range(budget):
        node  = rootnode
        state = rootstate.clone()
        node  = selection_phase(node, state, selection_policy=UCB1, selection_policy_args=[exploration_factor_ucb1])
        node  = expansion_phase(node, state)
        rollout_phase(state, rollout_budget)
        backpropagation_phase(node, state)

    return action_selection_phase(rootnode)
