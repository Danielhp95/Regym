from typing import List, Dict, Callable, Tuple
from math import sqrt

import random
import gym
from regym.rl_algorithms.MCTS.util import UCB1
from regym.rl_algorithms.MCTS.simultaneous_open_loop_node import SimultaneousOpenLoopNode


def selection_phase(nodes: List, state: gym.Env,
                    selection_policy: Callable[[object], float] = UCB1,
                    selection_policy_args: List = []) -> List:
    '''
    This function joins the selection and expansion phase of the vanilla
    MCTS algorithm. It begins by descending all trees in :param: nodes
    (One for each player) simultaneously, updating the :param: state with
    an action from each player. On already expanded nodes, the tree is
    descended according to :param: selection_policy. Due to the fact
    that trees can be asymmetrical, a tree might have reached a leaf node
    whilst another is still traversing down an already expanded branch.
    If this case is reached, the tree whose branch has been reached will
    start expanding nodes until all trees in :param: nodes reach a leaf.

    :param nodes: Trees to be descended and expanded
    :param state: Environment state, which will be modified
    :param selection_policy: function used to select a node from a given set of child nodes
    :params selection_policy_args: Parameters for :param selection_policy function
    :returns: List of nodes, where each node corresponds to the last
              expanded node on each player's tree.
    '''
    expanded = [False for _ in nodes]
    while not (all(expanded) or state.is_over()):
        moves, expanded = choose_moves(nodes, selection_policy, selection_policy_args)
        state.step(moves)
        nodes = [n.descend_and_expand(moves, state) for n in nodes]
    return nodes


def choose_moves(nodes: List,
                 selection_policy: Callable[[object], float],
                 selection_policy_args: Dict) -> Tuple[List[object], List[bool]]:
    '''
    Selects a move for each node in :param: nodes. Depending on whether
    the node is expanded or not the :param: selection_policy is used to
    select a move. Otherwise a random move is selected
    (i.e expansion strategy is random expansion).
    :param nodes: Nodes for which an action will be selected
    :param selection_policy: function used to select a node from a given set of child nodes
    :params selection_policy_args: Parameters for :param selection_policy function
    :returns: A list of selected moves, alongside a list stating which tree
              in :param: nodes should be expanded.
    '''
    expanded, moves = [], []
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


def rollout_phase(state: gym.Env):
    # TODO: Currently we just stop once we reach the end of the tree
    # TODO: Implement random acting (adapt from sequential mcts)
    return state


def backpropagation_phase(nodes: List, state: gym.Env) -> None:
    '''
    Updates the statistics of each player by propagating the results
    of the rollout_phase phase on each node in :params: node,
    ascending through each tree. Each tree's statistics are updated
    with respect to the perspective_player of each player.

    :param nodes: Nodes to be updated with the results of the rollout_phase
    :param state: Environment state at the end of rollout_phase
    '''
    for n in nodes:
        while n is not None:
            n.update(state.get_result(n.perspective_player))
            n = n.parent_node


def action_selection_phase(nodes: List) -> List[int]:
    '''
    Selects an action from each root node in :param: nodes based
    on a selection strategy.
    The selection strategy is: Choose node with highest expected payoff.
    :param nodes: Root nodes for each player's trees
    :returns: Best move for each node in :param: nodes according
              to selection strategy.
    '''
    return [sorted(n.child_nodes, key=lambda c: c.wins / c.visits)[-1].move
            for n in nodes]


def MCTS_UCT(rootstate, itermax: int, num_agents: int,
             exploration_factor_ucb1: float = sqrt(2)) -> int:
    '''
    Conducts a game tree search using the MCTS-UCT algorithm
    for a total of :param: itermax iterations using an open loop approach
    for SIMULTANEOUS multiagent environments. The search begins in the
    :param: rootstate.

    This algorithm updates :param: num_agents trees, one for each player.
    It technically computes a distribution over actions for each player,
    but we only :return: the action for the agent that called this function
    (which is assumed to be the 0th player)

    ASSUMPTION: All other agents use this version of MCTS. This could later
    be extended via opponent modelling.

    :param rootstate: The game state for which an action must be selected.
    :param itermax: number of MCTS iterations to be carried out.
                    Also knwon as the computational budget.
    :param exploration_factor_ucb1: 'c' constant in UCB1 equation.
    :returns: Action to be taken by player
    '''
    root_nodes = [SimultaneousOpenLoopNode(state=rootstate,
                                           perspective_player=i)
                  for i in range(num_agents)]

    for _ in range(itermax):
        nodes = root_nodes
        state = rootstate.clone()
        nodes = selection_phase(nodes, state, selection_policy=UCB1, selection_policy_args=[exploration_factor_ucb1])
        rollout_phase(state)
        backpropagation_phase(nodes, state)

    all_player_actions = action_selection_phase(root_nodes)
    return all_player_actions[0]  # TODO: this might be problematic. Look into it.
