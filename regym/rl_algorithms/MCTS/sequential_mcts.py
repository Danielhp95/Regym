from typing import List, Callable, Optional
from copy import deepcopy
from math import sqrt
import random

import numpy as np
import gym

from .util import UCB1, PUCT
from .sequential_open_loop_node import SequentialOpenLoopNode


def selection_phase(node: SequentialOpenLoopNode, state: gym.Env,
                    policy_fn: Callable,
                    selection_strat: Callable, selection_strat_args):
    '''
    TODO:
    '''
    if not node.is_fully_expanded() or node.child_nodes == []:
        return node
    selected_node = max(node.child_nodes,
                        key=lambda child: selection_strat(node, child, *selection_strat_args))
    state.step(selected_node.move)
    return selection_phase(selected_node, state, policy_fn,
                           selection_strat, selection_strat_args)


def expansion_phase_random_child(node, state, policy_fn: Callable):
    '''
    TODO
    :param policy_fn: Ignored, left here tor interface consistency
    '''
    if node.untried_moves != []:  # if we can expand (i.e. state/node is non-terminal)
        move = random.choice(node.untried_moves)
        state.step(move)
        node = node.add_child(move, state)
    return node


# Watchout for already initialized child nodes
def expansion_phase_expand_all(node, state, policy_fn: Callable):
    if node.untried_moves != []:  # if we can expand (i.e. state/node is non-terminal)
        legal_actions = state.get_moves()
        priors = policy_fn(state, legal_actions)
        for move, prior in zip(legal_actions, priors):
            state_clone = deepcopy(state)  # This operation might be slow
            state_clone.step(move)
            node.add_child(move, state_clone, prior)
        sampled_child_move = np.random.choice(legal_actions, p=priors)
        state.step(sampled_child_move)
        return next(n for n in node.child_nodes if n.move == sampled_child_move)
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
    visited_nodes = filter(lambda n: n.visits > 0, node.child_nodes)
    return sorted(visited_nodes, key=lambda c: c.wins / c.visits)[-1].move


def MCTS_UCT(rootstate, budget: int, num_agents: int,
             player_index: int,
             selection_strat: str,
             rollout_budget=100000,  # Large value -> unbounded rollout_phase
             policy_fn: Callable[[object, Optional[object]], List[float]] = None,
             exploration_factor: float = 2.5):
    """
    Conducts a game tree search using the MCTS-UCT algorithm
    for a total of param itermax iterations. The search begins
    in the param rootstate. Assumes that 2 players are alternating
    with results being [0.0, 1.0].

    :param rootstate: The game state for which an action must be selected.
    :param budget: number of MCTS iterations to be carried out. Also knwon as the computational budget.
    :param player_index: Player whom called this algorithm
    :param num_agents: UNUSED

    :param policy_fn: Policy which generates priors for each node
                      to be selected in PUCT. TODO: further explain

    :param exploration_factor_puct: Exploration constant to be used during
                                    PUCT calculations.
    :returns: (int) Action that will be taken by an agent.
    """
    if selection_strat   == 'ucb1':
        selection_policy, expansion_phase = UCB1, expansion_phase_random_child
    elif selection_strat == 'puct':
        selection_policy, expansion_phase = PUCT, expansion_phase_expand_all
    else: raise ValueError(f'Unknown :param: selection strat: {selection_strat}')

    rootnode = SequentialOpenLoopNode(state=rootstate)

    for _ in range(budget):
        node  = rootnode
        state = rootstate.clone()
        node  = selection_phase(node, state, policy_fn,
                                selection_strat=selection_policy,
                                selection_strat_args=[exploration_factor])
        node  = expansion_phase(node, state, policy_fn=policy_fn)
        rollout_phase(state, rollout_budget)
        backpropagation_phase(node, state)

    action = action_selection_phase(rootnode)
    child_visitations = {n.move: n.visits for n in rootnode.child_nodes}
    return action, child_visitations
