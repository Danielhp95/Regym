from typing import List, Callable
from copy import deepcopy
import random

import gym

from .util import extract_best_actions, add_dirichlet_noise
from .new_sequential_open_loop_node import Node


def get_actions_and_priors(state: gym.Env, obs, policy_fn):
    actions = state.get_moves()
    priors = policy_fn(obs, actions)
    P_a = {a_i: priors[a_i] for a_i in actions}
    return actions, P_a


def selection_phase(node, state: gym.Env, selection_strat, exploration_factor):
    if node.is_terminal:
        return node, None
    scores = selection_strat(node, exploration_factor)
    best_a_i = random.choice(extract_best_actions(scores))
    if best_a_i not in node.children:
        return node, best_a_i
    else:
        state.step(best_a_i)
        return selection_phase(node.children[best_a_i], state,
                               selection_strat, exploration_factor)


def expansion_phase_expand_all(node: Node, a_i: int, state: gym.Env,
                               policy_fn, player: int):
    observations, _, _, _ = state.step(a_i)
    actions, priors = get_actions_and_priors(state, observations[player],
                                             policy_fn)
    node.add_child(a=a_i, P_a=priors, actions=actions, player=player)
    return node.children[a_i], observations  # NOTE: observations are ignored for now


def backpropagation_phase(node: Node, value: float):
    node.N += 1
    if node.parent is None: return
    node.parent.update(a_i=node.a, value=value)
    backpropagation_phase(node.parent, -value)  # ASSUMPTION: 2-player zero sum game


def action_selection_phase(node: Node):
    return max(node.N_a, key=lambda x: node.N_a.get(x))


def rollout_phase(state: gym.Env, node: Node, obs, rollout_budget: int,
                  evaluation_fn: Callable[[object, List[int]], float]):
    rewards = None
    for i in range(rollout_budget):
        moves = state.get_moves()
        if moves == []: break
        obs, rewards, _, _ = state.step(random.choice(state.get_moves()))
    # We will need to put this in tensor
    if evaluation_fn: value = evaluation_fn(obs[node.player], state.get_moves())
    else: value = state.get_result(node.player)
    return value


def MCTS(rootstate, observation,
         budget: int,
         rollout_budget: int,
         selection_strat: Callable,
         exploration_factor: float,
         player_index: int,
         policy_fn: Callable,
         evaluation_fn: Callable,
         use_dirichlet: bool,
         dirichlet_alpha: float,
         num_agents: int):

    actions, priors = get_actions_and_priors(rootstate, observation, policy_fn)
    if use_dirichlet:
        priors = add_dirichlet_noise(alpha=dirichlet_alpha, p=priors)
    rootnode = Node(parent=None, player=player_index, a='R', actions=actions, priors=priors)

    for _ in range(budget):
        node = rootnode
        state = deepcopy(rootstate)
        node, a_i = selection_phase(node, state, selection_strat, exploration_factor)
        if not node.is_terminal:
            node, obs = expansion_phase_expand_all(node, a_i, state, policy_fn,
                                                   player=((node.player + 1) % num_agents))
            value = rollout_phase(state, node, obs, rollout_budget, evaluation_fn)
        backpropagation_phase(node, -1 * value)
        # Remember to maybe multiple by -1

    best_action = action_selection_phase(rootnode)
    return best_action, rootnode.N_a
