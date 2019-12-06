from typing import List
import gym

import numpy as np

from regym.environments import Task, EnvType
from regym.util import play_multiple_matches


def compute_winrate_matrix_metagame(population: List,
                                    episodes_per_matchup: int,
                                    task: Task,
                                    num_workers: int = 1) -> np.ndarray:
    '''
    Generates a metagame for a multiagent :param: task given a :param: population
    of strategies. This metagame is a symmetric 2-player zero-sum normal form game.
    The game is defined by a single matrix corresponding to the payoffs of player 1.
    The payoff matrix is square, it's shape is (rows=len(population), columns=len(population))
    The payoffs (Each entry [i,j] in the metagame matrix) are computed by computing
    the agent i's winrate against agent j for :param: episodes_per_matchup.
    Assumptions:
        - Game is symmetric. Strategy 1 has the same chances of winning
          against strategy 2 regardless of whether player 1 chooses strategy 1
          or player 2 chooses strategy 1. The payoff of a given strategy is not
          affected by player identity.

    TODO future improvement:
       In non-deterministic games (every game, because our policies are stochastic)
       in order to compute a winrate for a certain matchup we need to pit two
       agents against each other for a certain number of episodes. Instead of
       exhaustively playing each matchup :param: episodes_per_matchup, we can
       think of computing the winrate for entry [agent1, agent2] as a bandit
       problem. This idea is expanded in: https://arxiv.org/abs/1909.09849

    :param population: List of agents which will be pitted against each other to generate
                       a metagame for the given :param: task.
    :param episodes_per_matchup: Number of times each matchup will be repeated to compute
                                 empirical winrates. Higher values generate a more accurate
                                 metagame, at the expense of longer compute time.
    :param task Multiagent Task for which the metagame is being computed
    :param num_workers: Number of parallel threads to be spawned to carry out benchmarks in parallel
    :returns: Empirical payoff matrix for player 1 representing the metagame for :param: task and
              :param: population
    '''
    check_input_validity(population, episodes_per_matchup, task)

    upper_triangular_winrate_matrix = generate_upper_triangular_symmetric_metagame(population, task, episodes_per_matchup, num_workers)

    # Copy upper triangular into lower triangular  Generate complementary entries
    # a_i,j + a_j,i = 1 for all non diagonal entries
    winrate_matrix = upper_triangular_winrate_matrix + \
                     (np.triu(np.ones_like(upper_triangular_winrate_matrix), k=1)
                      - upper_triangular_winrate_matrix).transpose()
    for i in range(len(winrate_matrix)): winrate_matrix[i, i] = 0.5 # Generate diagonal of 0.5
    return winrate_matrix


def generate_upper_triangular_symmetric_metagame(population: List, task: Task, episodes_per_matchup: int,
                                                 num_workers: int = 1) -> np.ndarray:
    '''
    Generates a matrix which:
        - Upper triangular part contains the empirical winrates of pitting each agent in
          :param: population against each other in :param: task for :param: episodes_per_matchup
        - Diagonal and lower triangular parts of the matrix are filled with 0s.

    :param population: List of agents which will be pitted against each other to generate
                       a metagame for the given :param: task.
    :param episodes_per_matchup: Number of times each matchup will be repeated to compute
                                 empirical winrates. Higher values generate a more accurate
                                 metagame, at the expense of longer compute time.
    :param task Multiagent Task for which the metagame is being computed
    :param num_workers: Number of parallel threads to be spawned to carry out benchmarks in parallel
    :returns: PARTIALLY filled in payoff matrix for metagame for :param: population in :param: task.
    '''
    winrate_matrix = np.zeros((len(population), len(population)))
    # k=1 below makes sure that the diagonal indices are not included
    matchups_agent_indices = zip(*np.triu_indices_from(winrate_matrix, k=1))

    for i, j in matchups_agent_indices:
        player_1_winrate = play_multiple_matches(task,
                                                 agent_vector=(population[i], population[j]),
                                                 n_matches=episodes_per_matchup)[0]
        winrate_matrix[i, j] = player_1_winrate
    return winrate_matrix


def check_input_validity(population: np.ndarray, episodes_per_matchup: int, task):
    if population is None: raise ValueError('Population should be an array of policies')
    if len(population) == 0: raise ValueError('Population cannot be empty')
    if episodes_per_matchup <= 0: raise ValueError('Episodes_per_matchup must strictly positive')
    if task.env_type == EnvType.SINGLE_AGENT: raise ValueError('Task is single-agent. Metagames can only be computed for multiagent tasks.')
