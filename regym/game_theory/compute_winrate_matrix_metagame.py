from typing import List, Tuple
from itertools import product

from tqdm import tqdm
import numpy as np
import gym

from regym.rl_algorithms.agents import Agent
from regym.environments import Task, EnvType
from regym.util import play_multiple_matches
from regym.game_theory import solve_zero_sum_game
from regym.rl_loops import compute_winrates


def compute_winrate_matrix_metagame(population: List[Agent],
                                    episodes_per_matchup: int,
                                    task: Task,
                                    is_game_symmetrical: bool = False,
                                    num_envs: int = 1,
                                    show_progress: bool = False) -> np.ndarray:
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
    :param task: Multiagent Task for which the metagame is being computed
    :param num_envs: Number of parallel enviroments to spawn during
                     the computation of the underlying winrate matrix.
                     If -1 is specified, it will internally be transformed to
                     the cpu count.
    :returns: Empirical payoff matrix for player 1 representing the metagame for :param: task and
              :param: population
    '''
    check_input_validity(population, episodes_per_matchup, task, is_game_symmetrical)

    upper_triangular_winrate_matrix = generate_upper_triangular_symmetric_metagame(
        population,
        task,
        episodes_per_matchup,
        is_game_symmetrical,
        num_envs,
        show_progress)

    # Copy upper triangular into lower triangular  Generate complementary entries
    # a_i,j + a_j,i = 1 for all non diagonal entries
    winrate_matrix = upper_triangular_winrate_matrix + \
                     (np.triu(np.ones_like(upper_triangular_winrate_matrix), k=1)
                      - upper_triangular_winrate_matrix).transpose()
    for i in range(len(winrate_matrix)): winrate_matrix[i, i] = 0.5 # Generate diagonal of 0.5
    return winrate_matrix


def generate_upper_triangular_symmetric_metagame(population: List[Agent],
                                                 task: Task,
                                                 episodes_per_matchup: int,
                                                 is_game_symmetrical: bool=False,
                                                 num_envs: int = 1,
                                                 show_progress: bool=False) -> np.ndarray:
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
    :param is_game_symmetrical: If true, player positions will not be shuffled. (TODO: improve)
    :param task Multiagent Task for which the metagame is being computed
    :param num_envs: Number of parallel enviroments to spawn during
                     the computation of the underlying winrate matrix.
                     If -1 is specified, it will internally be transformed to
                     the cpu count.
    :returns: PARTIALLY filled in payoff matrix for metagame for :param: population in :param: task.
    '''
    winrate_matrix = np.zeros((len(population), len(population)))
    # k=1 below makes sure that the diagonal indices are not included
    matchups_agent_indices = list(zip(*np.triu_indices_from(winrate_matrix, k=1)))
    if show_progress:
        matchups_agent_indices = tqdm(
            matchups_agent_indices,
            desc=f"Computing evaluation matrix winrate metagame, num_envs={num_envs}, episodes_per_matchup={episodes_per_matchup} on task {task.name}",
            total=len(matchups_agent_indices)
        )

    for i, j in matchups_agent_indices:
        # TODO: refactor smartly into a single function
        if is_game_symmetrical:
            winrates = compute_winrates_in_parallel(
                population[i], population[j],
                task, episodes_per_matchup, num_envs)
        else:
            winrates_1 = compute_winrates_in_parallel(
                population[i], population[j],
                task,
                episodes_per_matchup // 2,
                num_envs)
            winrates_2 = compute_winrates_in_parallel(
                population[j], population[i],
                task,
                episodes_per_matchup // 2,
                num_envs)[::-1]  # We reverse the winrates so that agent i's winrate is first
            winrates = (winrates_1 + winrates_2) / 2
        winrate_matrix[i, j] = winrates[0]
    return winrate_matrix


def compute_winrates_in_parallel(agent_1: Agent, agent_2: Agent,
                                 task: Task,
                                 num_episodes: int, num_envs: int) -> np.ndarray:
    trajectories = task.run_episodes(
        agent_vector=[agent_1, agent_2],
        num_episodes=num_episodes,
        num_envs=num_envs,
        training=False
    )
    winrates = [len(list(filter(lambda t: t.winner == a_i, trajectories))) / len(trajectories)
                for a_i in range(2)]  # TODO, improve, instead of using magic number
    return np.array(winrates)


def generate_evaluation_matrix_multi_population(populations: List[List[Agent]],
                                                task: Task,
                                                episodes_per_matchup: int,
                                                num_envs: int = 1,
                                                show_progress: bool = True) -> np.ndarray:
    '''
    Generates an evaluation matrix (a metagame) for a multiagent :param: task
    given a set of :param: populations, each containing a (possibly uneven) number
    of Agents (strategies). This metagame is an n-player zero-sum normal form game,
    where `n = len(populations)`.

    Currently only 2 populations are suported

    The game is defined by a matrix single matrix, of shape: (rows=len(population[0]), columns=len(population[1]))
    The payoffs (Each entry [i,j] in the evaluation metagame matrix) represent
    agent i's (from populations[0]) winrate against agent j (from popuations[1])
    for :param: episodes_per_matchup.

    If all populations contain exactly the same agents, this function will
    (theoretically) yield the same result as `compute_winrate_matrix_metagame(populations[0],...)`

    :param populations: Populations to be matched against each other
    :param task: Multiagent Task for which the evaluation matrix is being computed
    :param episodes_per_matchup: Number of times each matchup will be repeated to compute
                                 empirical winrates. Higher values generate a more accurate
                                 metagame, at the expense of longer compute time.
    :param num_envs: Number of parallel enviroments to spawn during
                     the computation of the underlying winrate matrix.
                     If -1 is specified, it will internally be transformed to
                     the cpu count.
    :param show_progress: Whether to show a progress bar to stdout
    :returns: Emprirical winrate matrix (aka evaluation matrix) representing
              the winrates of populations[0] against population[1]. That is:
              each row i represents the winrates of agent i from popuations[0]
              against all agents from populations[1]
    '''
    if task.env_type == EnvType.SINGLE_AGENT:
        raise ValueError('Task is single-agent. Metagames can only be computed for multiagent tasks.')
    if len(populations) > 2:
        raise NotImplementedError('Currently only two popuations are supported')
    if num_envs < 1 and num_envs != -1:
        raise ValueError(f'Param `num_envs` has to be equal or greater than 1, unless -1 is specified, which is internally changed to the cpu count. Given {num_envs}')

    if num_envs == 1:
        winrate_matrix = multi_population_winrate_matrix_computation(populations, task, episodes_per_matchup)
    else:
        if task.env_type == EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
            winrate_matrix = parallel_multi_population_winrate_matrix_computation(
                populations,
                task,
                episodes_per_matchup,
                num_envs,
                show_progress
            )
        else:  # EnvType.MULTIAGENT_SEQUENTIAL_ACTION:
            '''
            HACK: Currently, shuffling agent positions is not supported
            parallel environments, so we run half the episodes on one position, and half on the other
            '''
            winrate_matrix_1 = parallel_multi_population_winrate_matrix_computation(
                [populations[0], populations[1]],
                task,
                episodes_per_matchup // 2,
                num_envs,
                show_progress
            )
            winrate_matrix_2 = parallel_multi_population_winrate_matrix_computation(
                [populations[1], populations[0]],
                task,
                episodes_per_matchup // 2,
                num_envs,
                show_progress
            )
            # Transposing is necessary in case that both populations
            winrate_matrix = (winrate_matrix_1 + winrate_matrix_2.transpose()) / 2
    return winrate_matrix


# TODO: possibly refactor this and function below into single function.
def multi_population_winrate_matrix_computation(populations: List[List[Agent]],
                                                task: Task,
                                                episodes_per_matchup: int) -> np.ndarray:
    '''
    TODO: Check wether removing this function is safe
    '''
    population_1, population_2 = populations
    winrate_matrix = np.zeros((len(population_1), len(population_2)))
    for i, j in product(range(len(population_1)), range(len(population_2))):
        player_1_winrate = play_multiple_matches(
            task,
            agent_vector=[
                population_1[i],
                population_2[j]
            ],
            n_matches=episodes_per_matchup
        )[0]
        winrate_matrix[i, j] = player_1_winrate
    return winrate_matrix


def parallel_multi_population_winrate_matrix_computation(populations: List[List[Agent]],
                                                         task: Task,
                                                         episodes_per_matchup: int,
                                                         num_envs: int,
                                                         show_progress: bool = False) -> np.ndarray:
    '''
    TODO
    '''
    population_1, population_2 = populations
    winrate_matrix = np.zeros((len(population_1), len(population_2)))
    matchups_agent_indices = list(product(range(len(population_1)), range(len(population_2))))
    if show_progress:
        matchups_agent_indices = tqdm(
            matchups_agent_indices,
            desc=f"Computing multi population winrate matrix, num_envs={num_envs}, episodes_per_matchup={episodes_per_matchup} on task {task.name}",
            total=len(matchups_agent_indices)
        )
    for i, j in matchups_agent_indices:
        # We compute 2 sets of trjectories, one where population_1 plays first and another where population_2 plays first
        trajectories_1 = task.run_episodes(
            agent_vector=[
                population_1[i],
                population_2[j]
            ],
            num_episodes=episodes_per_matchup // 2,
            num_envs=num_envs,
            training=False
        )
        trajectories_2 = task.run_episodes(
            agent_vector=[
                population_2[j],
                population_1[i]
            ],
            num_episodes=episodes_per_matchup // 2,
            num_envs=num_envs,
            training=False
        )
        winrates_1 = compute_winrates(trajectories_1)
        winrates_2 = compute_winrates(trajectories_2)
        winrate_matrix[i, j] = (winrates_1[0] + winrates_2[1]) / 2
    return winrate_matrix


def relative_population_performance(population_1: List[Agent],
                                    population_2: List[Agent],
                                    task: Task,
                                    num_envs: int=-1,
                                    episodes_per_matchup: int=100) -> Tuple[float, np.ndarray]:
    '''
    From 'Open Ended Learning in Symmetric Zero-sum Games'
    https://arxiv.org/abs/1901.08106
    Definition 3: Relative population performance.
    Computes an scalar value comparing the performance of :param: population_1
    against :param: population_2 at :param: task.

    It boils down to computing the value of player 1 under Nash Equilibrium
    of a zero-sum game computed from matching :param: population_1 (player 1)
    against :param: population_2 (player 2) inside of :param: task.

    Proposition 5.1 from paper above. Relative population performance is
    independent of choice of Nash Equilibrium.
    :param population_1 / _2: Populations from which relative population
                              performance will be computed
    :param task: Multiagent Task for which the metagame is being computed
    :param num_envs: Number of parallel environments to set up to compute winrate matrices
    :param episodes_per_matchup: Number of times each matchup will be repeated to compute
                                 empirical winrates. Higher values generate a more accurate
                                 metagame, at the expense of longer compute time.
    :returns:
        - Population performance of :param: population_1 relative to
          :param: population_2.
        - Empirical winrate metagame normal form game.
    '''
    relative_performance_evolution, final_winrate_metagame = evolution_relative_population_performance(
        population_1, population_2,
        task,
        episodes_per_matchup,
        num_envs=num_envs,
        initial_index=(len(population_1) -1))
    return relative_performance_evolution[0], final_winrate_metagame


def evolution_relative_population_performance(population_1: List[Agent],
                                              population_2: List[Agent],
                                              task: Task,
                                              episodes_per_matchup: int,
                                              num_envs: int = -1,
                                              initial_index: int = 0) \
                                              -> Tuple[np.ndarray, np.ndarray]:
    '''
    Computes various relative population performances for :param: population_1
    and :param: population_2, where the first relative population performance
    is taken for population_1[0:initial_index] policies, until
    population_1[:], with a step of 1.

    Useful for plotting relative population performance as new policies
    were introduced in both populations.

    :param population_1 / _2: Populations from which relative population
                              performance will be computed
    :param task: Multiagent Task for which the metagame is being computed
    :param episodes_per_matchup: Number of times each matchup will be repeated to compute
                                 empirical winrates. Higher values generate a more accurate
                                 metagame, at the expense of longer compute time.
    :param initial_index: Index for both populations at which the relative
                          population performance will be computed.
    :param num_envs: Number of parallel enviroments to spawn during
                     the computation of the underlying winrate matrix.
                     If -1 is specified, it will internally be transformed to
                     the cpu count.
    :returns: Vector containing the evolution of the population performance of
              :param: population_1 relative to :param: population_2 starting
              at population_1 index :param: initial_index.
    '''

    if task.env_type == EnvType.SINGLE_AGENT:
        raise ValueError('Task is single-agent. Metagames can only be computed for multiagent tasks.')
    if len(population_1) != len(population_2):
        raise ValueError(f'Population must be of the same size: size 1 {len(population_1)}, size 2 {len(population_2)}. This constraint could be relaxed')
    if not (0 <= initial_index < len(population_1)):
        raise ValueError(f'Initial index must be a valid index for population vector: [0,{len(population_1)}]')
    if episodes_per_matchup <= 0:
        raise ValueError(f'Param `episodes_per_matchup` must be strictly positive')
    if num_envs < 1 and num_envs != -1:
        raise ValueError(f'Param `num_envs` has to be equal or greater than 1, unless -1 is specified, which is internally changed to the cpu count. Given {num_envs}')

    winrate_matrix_metagame = generate_evaluation_matrix_multi_population(populations=[
                                                                     population_1,
                                                                     population_2
                                                                     ],
                                                                 task=task,
                                                                 episodes_per_matchup=episodes_per_matchup,
                                                                 num_envs=num_envs)
    # The antisymmetry refers to the operation performed to the winrates inside
    # of the matrix, NOT the matrix itself
    antisymmetric_form = winrate_matrix_metagame - 1/2
    relative_performances = np.zeros(len(population_1) - initial_index)
    for i in range(initial_index + 1, len(population_1) + 1):
        _, _, value_1, _ = solve_zero_sum_game(antisymmetric_form[:i, :i])
        relative_performances[(i-1) - initial_index] = value_1
    return relative_performances, winrate_matrix_metagame


def check_input_validity(population: List[Agent], episodes_per_matchup: int, task: Task, is_game_symmetrical: bool):
    if population is None: raise ValueError('Population should be an array of policies')
    if len(population) == 0: raise ValueError('Population cannot be empty')
    if episodes_per_matchup <= 0: raise ValueError('Episodes_per_matchup must strictly positive')
    if task.env_type == EnvType.SINGLE_AGENT: raise ValueError('Task is single-agent. Metagames can only be computed for multiagent tasks.')
    if not isinstance(is_game_symmetrical, bool): raise ValueError(f'Param "is_game_symmetrical" must be a bool, found {type(is_game_symmetrical)}')
