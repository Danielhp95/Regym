from typing import List, Callable, Any, Tuple, Dict, Optional, Union
from copy import deepcopy
from random import choice

import gym

from .util import extract_best_actions, add_dirichlet_noise
from .sequential_node import SequentialNode


def get_actions_and_priors(state: gym.Env,
                           observation,
                           policy_fn: Callable[[Any, List[int]], List[float]],
                           player_index: int) -> Tuple[List[int], Dict[int, float]]:
    '''
    Extracts:
    - Available legal actions at :param: state
    - Priors for each legal action by calling :param: policy_fn

    :param state: Environment model
    :param obs: Observation corresponding to the player at current node
    :param policy_fn: Policy which generates priors for each node
                      to be selected in PUCT.
    :returns: Available actions at :param: state and priors over them
    '''
    actions = state.get_moves()
    priors = policy_fn(observation, actions, requested_player_index=player_index)
    P_a = {a_i: priors[a_i] for a_i in actions}
    return actions, P_a


def selection_phase(node: SequentialNode, state: gym.Env,
                    selection_strat: Callable[[SequentialNode, float], Dict[int, float]],
                    exploration_factor: float) \
                    -> Tuple[SequentialNode, Union[int, None]]:
    '''
    Traverses the tree starting at :param: node, following
    :param: selection_strat to decide which child (branch) to follow, updating
    the :param state environment model during traversal. This phase terminates
    when a child node (edge) is selected leading to an un-expanded child node.
    OR when a terminal node is found.

    :param node: Current node of the MCTS tree being traversed
    :param state: Environment model
    :param selection_strat: Selection policy used to select the most promising
                            child node to traverse to
    :param exploration_factor: Exploration factor for :param: selection_strat
    :returns: A node and an action leading to an un-expanded child
    '''
    if node.is_terminal:
        return node, None
    scores = selection_strat(node, exploration_factor)
    best_a_i = choice(extract_best_actions(scores))
    if best_a_i not in node.children:
        return node, best_a_i
    else:
        state.step(best_a_i)
        return selection_phase(node.children[best_a_i], state,
                               selection_strat, exploration_factor)


def expansion_phase_expand_all(node: SequentialNode, a_i: int, state: gym.Env,
                               policy_fn, player_index: int) -> Tuple[SequentialNode, Any]:
    '''
    Expands a child node from :param: node with corresponding
    action :param: a_i, which is used to `step` the environment model
    :param: state.

    :param node: Node whose child is going to be expanded
    :param a_i: Action to be taken in the environment, branching from
                :param: node
    :param player: Player at the node which is to be expanded
    :returns: Newly expanded node and the observations for all players
              in the environment after action :param: a_i was taken
    '''
    observations, _, _, _ = state.step(a_i)
    actions, priors = get_actions_and_priors(state, observations[player_index],
                                             policy_fn,
                                             player_index)
    node.add_child(a=a_i, P_a=priors, actions=actions, player=player_index)
    # NOTE: observations are currently ignored
    return node.children[a_i], observations


def backpropagation_phase(node: SequentialNode, value: float):
    '''
    Backpropagates :param: value up the MCTS tree starting at :param: node.
    Because we assume 2-player zero sum game, we negate :param: value
    every time we go up one node.

    :param node: Current node in the backpropagation traversal
    :param value: Value to backpropagate to the :param: node's parent.
    '''
    node.N += 1
    if node.parent is None: return
    node.parent.update(a_i=node.a, value=value)
    backpropagation_phase(node.parent, -value)  # ASSUMPTION: 2-player zero sum game


def action_selection_phase(node: SequentialNode) -> int:
    '''
    Selects an action id (int) from :param: node corresponding to the "best"
    child under the following this criteria:
    - Child node with highest visit count
    :param Node: Node from which an action is going to be selected
    :returns: Action id corresponding to the "best" action
              following criteria above
    '''
    return max(node.N_a, key=lambda x: node.N_a.get(x))


def rollout_phase(state: gym.Env, node: SequentialNode, observations: List,
                  rollout_budget: int,
                  evaluation_fn: Optional[Callable[[Any, List[int]], float]]) \
                  -> float:
    '''
    Starts a rollout at :param: state following a RANDOM policy,
    which continues until :param: rollout_budget actions have been taken
    or a terminal node is found. It returns a scalar (float) value from the
    perspective of :param: node. If :param: evaluation_fn is present,
    it will be used to compute the value to backpropagate. Otherwise it will
    be queried from the environment

    :param state: Environment model
    :param node: Leaf node in current MCTS tree
    :param observations: List of current observations for all agents
    :param evaluation_fn: Function used to compute a value to backpropagate
    :returns: Value to backpropagate from the perspective of the player
              in :param: node
    '''
    for i in range(rollout_budget):
        moves = state.get_moves()
        if moves == []:
            break
        observations, _, _, _ = state.step(choice(state.get_moves()))
    if state.get_moves() == [] or not evaluation_fn:
        value = state.get_result(node.player)
    else:
        value = evaluation_fn(observations[node.player], state.get_moves())
    return value


def MCTS(rootstate: gym.Env, observation,
         budget: int,
         rollout_budget: int,
         selection_strat: Callable,
         exploration_factor: float,
         player_index: int,
         policy_fn: Callable[[Any, List[int]], List[float]],
         evaluation_fn: Optional[Callable[[Any, List[int]], float]],
         use_dirichlet: bool,
         dirichlet_alpha: float,
         num_agents: int) -> Tuple[int, Dict[int, int]]:
    '''
    Conducts an MCTS game tree search where the root node represents
    :param: rootstate with :param: observation for the :param: player_index.
    MCTS will be carried out for a total of :param: budget iterations.

    If :param: use_dirichlet is set, the priors for the root node will be
    perturbed with Dirichlet noise $d ~ Dirichlet(\alpha)$ parameterized
    by :param: dirichlet_alpha, to encourage exploration.

    ASSUMPTIONS:
        - Game is 2 player zero-sum game
        - Turns are taken in clockwise fashion:
              P1 acts, P2 acts..., P :param: num_agents acts, P1 acts...

    Selection phase:
        - The selection strategy is defined by :param: selection_strat and
          parameterized by :param: exploration_factor

    Expansion phase:
        - All child nodes are expanded

    Rollout phase:
        - RANDOM actions are carried out for at most :param: rollout_budget
          and :param: policy_fn is used to compute a value to backpropagate,
          if not present, the resulting game state will be queried for a value

    Backpropagation phase:
        - All visited nodes are updated

    :param rootstate: The environment for which MCTS will compute
                      an approximate optimal action.
    :param observation: Observation for agent :param: player_index
                        at environment state :param: rootstate
    :param budget: Number of MCTS iterations to be carried out.
                   Also knwon as the computational budget
    :param rollout_budget: Number of steps to simulate during rollout phase
    :param selection_strat: Selection strategy used to select the best
                            child node to traverse to during selection phase
    :param exploration_factor: Exploration constant to be used during
                               seletion phase in :param: selection_strat
    :param player_index: Index of the agent in the :param: rootstate
                         environment's agent vector who invoked this function
    :param policy_fn: Function used to compute priors for each node.
                      Used as part of the PUCT selection strategy.
    :param evaluation_fn: Function used to compute a value to backpropagate.
                          Used at the end of the rollout phase
    :param use_dirichlet: Whether to add dirichlet noise to the priors of the
                          root node of the MCTS tree to encourage exploration
    :param dirichlet_alpha: Parameter of Dirichlet distribution.
                            Only used if :param: use_dirichlet flag is set
    :param num_agents: Number of agents present in the environment
    :returns: (int) Action to be taken by agent :param: player_index
              and visitations of each child in the root node
    '''

    actions, priors = get_actions_and_priors(rootstate, observation, policy_fn, player_index)
    if use_dirichlet:
        priors = add_dirichlet_noise(alpha=dirichlet_alpha, p=priors)
    rootnode = SequentialNode(parent=None, player=player_index, a='R',
                              actions=actions, priors=priors)

    for _ in range(budget):
        node = rootnode
        state = deepcopy(rootstate)
        node, a_i = selection_phase(node, state, selection_strat, exploration_factor)
        if not node.is_terminal:
            node, obs = expansion_phase_expand_all(node, a_i, state, policy_fn,
                                                   player_index=((node.player + 1) % num_agents))
            value = rollout_phase(state, node, obs,
                                  rollout_budget, evaluation_fn)
        # Multiply by -1 because 'node.parent' tries to minimize 'node' reward
        backpropagation_phase(node, -1 * value)

    best_action = action_selection_phase(rootnode)
    return best_action, rootnode.N_a
