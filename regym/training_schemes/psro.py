'''
Implementation of Policy-Spaced Response Oracles (PSRO) as first introduced in Lanctot et al 2017:
http://papers.nips.cc/paper/7007-a-unified-game-theoretic-approach-to-multiagent-reinforcement-learning

TODO: difference between PSRO which takes 3 separate stages and our method, which is an online method.
'''
from typing import Callable, List
from copy import deepcopy
import dill
import logging
import time
from itertools import product

import torch
import numpy as np

from regym.game_theory import solve_zero_sum_game
from regym.util import play_multiple_matches
from regym.environments import generate_task, Task, EnvType
from regym.rl_loops import Trajectory


def default_meta_game_solver(winrate_matrix: np.ndarray):
    return solve_zero_sum_game(winrate_matrix)[0].reshape((-1))


class PSRONashResponse():

    def __init__(self,
                 task: Task,
                 meta_game_solver: Callable = default_meta_game_solver,
                 threshold_best_response: float = 0.7,
                 benchmarking_episodes: int = 10,
                 match_outcome_rolling_window_size: int = 10):
        '''
        :param task: Multiagent task
        :param meta_game_solver: Function which takes a meta-game and returns a probability
                                 distribution over the policies in the meta-game.
                                 Default uses maxent-Nash equilibrium for the logodds transformation
                                 of the winrate_matrix metagame.
        :param threshold_best_response: Winrate thrshold after which the agent being
                                        trained is to converge towards a best response
                                        againts the current meta-game solution.
        :param benchmarking_episodes: Number of episodes that will be used to compute winrates
                                      to fill the metagame.
        :param match_outcome_rolling_window_size: Number of episodes that will be used to
                                                  decide whether the currently training agent
                                                  has converged to a best response.
        '''
        self.name = f'PSRO(M=maxentNash,O=BestResponse(wr={threshold_best_response},ws={match_outcome_rolling_window_size})'
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        self.check_parameter_validity(task, threshold_best_response,
                                      benchmarking_episodes,
                                      match_outcome_rolling_window_size)
        self.task = task

        self.meta_game_solver = meta_game_solver
        self.meta_game: np.ndarray = None
        self.meta_game_solution: np.ndarray = None
        self.menagerie = []

        self.threshold_best_response = threshold_best_response
        self.match_outcome_rolling_window = []
        self.match_outcome_rolling_window_size = match_outcome_rolling_window_size

        self.benchmarking_episodes = benchmarking_episodes

        self.statistics = [self.IterationStatistics(0, 0, 0, [0], np.nan)]

    def opponent_sampling_distribution(self, menagerie: List['Agent'], training_agent: 'Agent'):
        '''
        :param menagerie: archive of agents selected by the curator and the potential opponents
        :param training_agent: Agent currently being trained
        '''
        if len(menagerie) == 0 and len(self.menagerie) == 0:
            self.init_meta_game_and_solution(training_agent)
        sampled_index = np.random.choice([i for i in range(len(self.menagerie))],
                                         p=self.meta_game_solution)
        self.statistics[-1].menagerie_picks[sampled_index] += 1
        return [self.menagerie[sampled_index]]

    def init_meta_game_and_solution(self, training_agent):
        self.add_agent_to_menagerie(training_agent)
        self.meta_game = np.array([[0.5]])
        self.meta_game_solution = np.array([1.0])

    def curator(self, menagerie: List['Agent'], training_agent: 'Agent',
                episode_trajectory: Trajectory,
                training_agent_index: int,
                candidate_save_path: str) -> List['Agent']:
        '''
        :param menagerie: archive of agents selected by the curator and the potential opponents
        :param training_agent: Agent currently being trained
        :returns: menagerie to be used in the next training episode.
        '''
        self.statistics[-1].total_elapsed_episodes += 1
        self.statistics[-1].current_iteration_elapsed_episodes += 1

        self.update_rolling_winrates(episode_trajectory, training_agent_index)
        if self.has_policy_converged():
            self.add_agent_to_menagerie(training_agent, candidate_save_path)
            self.update_meta_game()
            self.update_meta_game_solution()
            self.match_outcome_rolling_window = []
            self.statistics += [self.create_new_iteration_statistics(self.statistics[-1])]
            self.statistics[-1].meta_game_solution = self.meta_game_solution
        return self.menagerie

    def has_policy_converged(self):
        current_winrate = (sum(self.match_outcome_rolling_window) \
                           / self.match_outcome_rolling_window_size)
        return current_winrate >= self.threshold_best_response

    def update_rolling_winrates(self, episode_trajectory: Trajectory, training_agent_index: int):
        victory = int(episode_trajectory.winner == training_agent_index)
        if len(self.match_outcome_rolling_window) >= self.match_outcome_rolling_window_size:
            self.match_outcome_rolling_window.pop(0)
        self.match_outcome_rolling_window.append(victory)

    def update_meta_game_solution(self, update=False):
        self.logger.info(f'START: Solving metagame. Size: {len(self.menagerie)}')
        start_time = time.time()
        self.meta_game_solution = self.meta_game_solver(self.meta_game)
        time_elapsed = time.time() - start_time
        self.statistics[-1].time_elapsed_meta_game_solution = time_elapsed
        self.logger.info(f'FINISH: Solving metagame. time: {time_elapsed}')
        return self.meta_game_solution

    def update_meta_game(self):
        self.logger.info(f'START: updating metagame. Size: {len(self.menagerie)}')
        start_time = time.time()
        number_old_policies = len(self.menagerie) - 1
        updated_meta_game = np.full(((len(self.menagerie)), len(self.menagerie)),
                                    np.nan)

        # Filling the matrix with already-known values.
        updated_meta_game[:number_old_policies, :number_old_policies] = self.meta_game

        self.fill_meta_game_missing_entries(self.menagerie, updated_meta_game, self.benchmarking_episodes,
                                            self.task)
        self.meta_game = updated_meta_game
        time_elapsed = time.time() - start_time
        self.statistics[-1].time_elapsed_meta_game_update = time_elapsed
        self.logger.info(f'FINISH: updating metagame. time: {time_elapsed}')
        return updated_meta_game

    def fill_meta_game_missing_entries(self, policies: List,
                                       updated_meta_game: np.ndarray,
                                       benchmarking_episodes: int, task: Task):
        indices_to_fill = product(range(updated_meta_game.shape[0]),
                                  [updated_meta_game.shape[0] - 1])
        for i, j in indices_to_fill:
            # TODO: maybe use regym.evaluation. benchmark on tasks?
            if i == j: updated_meta_game[j, j] = 0.5
            else:
                winrate_estimate = self.estimate_winrate(task, policies[i],
                                                         policies[j], benchmarking_episodes)
                # Because we are asumming a symmetrical zero-sum game.
                updated_meta_game[i, j] = winrate_estimate
                updated_meta_game[j, i] = 1 - winrate_estimate
        return updated_meta_game

    def estimate_winrate(self, task: Task, policy_1: 'Agent', policy_2: 'Agent',
                         benchmarking_episodes: int) -> float:
        '''
        ASSUMPTION: Task is a 2-player, non-symmetrical game.
        Thus the :param: policies need to be benchmarked on both positions.
        '''
        first_position_winrates = play_multiple_matches(
            task=task, agent_vector=[policy_1, policy_2],
            n_matches=(benchmarking_episodes // 2))
        second_position_winrates = play_multiple_matches(
            task=task, agent_vector=[policy_2, policy_1],
            n_matches=(benchmarking_episodes // 2))
        policy_1_overall_winrate = (first_position_winrates[0] +
                                    second_position_winrates[1]) / 2
        return policy_1_overall_winrate

    def add_agent_to_menagerie(self, training_agent, candidate_save_path=None):
        if candidate_save_path: torch.save(training_agent, candidate_save_path)
        cloned_agent = deepcopy(training_agent)
        cloned_agent.training_agent = False
        self.menagerie.append(cloned_agent)

    def create_new_iteration_statistics(self, last_iteration_statistics):
        return self.IterationStatistics(len(self.statistics), last_iteration_statistics.total_elapsed_episodes,
                                        0, [0] * len(self.menagerie),
                                        self.meta_game_solution)

    def check_parameter_validity(self, task, threshold_best_response,
                                 benchmarking_episodes,
                                 match_outcome_rolling_window_size):
        if task.env_type == EnvType.SINGLE_AGENT:
            raise ValueError('Task provided: {task.name} is singleagent. PSRO is a multiagent ' +
                             'meta algorithm. It only opperates on multiagent tasks')
        if not(0 <= threshold_best_response <= 1):
            raise ValueError('Parameter \'threshold_best_response\' represents ' +
                             'a winrate (a probability). It must lie between [0, 1]')
        if not(0 < benchmarking_episodes):
            raise ValueError('Parameter \'benchmarking_episodes\' must be strictly positive')
        if not(0 < match_outcome_rolling_window_size):
            raise ValueError('Parameter \'benchmarking_episodes\' corresponds to ' +
                             'the lenght of a list. It must be strictly positive')

    def __repr__(self):
        return f'{self.name}. Meta-game size: {len(self.menagerie)}.'

    class IterationStatistics():
        def __init__(self, iteration_number: int,
                     total_elapsed_episodes: int,
                     current_iteration_elapsed_episodes: int,
                     menagerie_picks: List[int],
                     meta_game_solution: np.ndarray):
            '''
            Data class containing information for each of the PSRO iterations
            '''
            self.iteration_number = iteration_number
            self.total_elapsed_episodes = total_elapsed_episodes
            self.current_iteration_elapsed_episodes = current_iteration_elapsed_episodes
            self.menagerie_picks = menagerie_picks
            self.meta_game_solution = meta_game_solution
            self.time_elapsed_meta_game_solution = np.nan
            self.time_elapsed_meta_game_update = np.nan

        def __repr__(self):
            s = \
            f'''
            Iteration: {self.iteration_number}
            Total elapsed episodes: {self.total_elapsed_episodes}
            Current iteration elapsed episodes: {self.current_iteration_elapsed_episodes}
            Menagerie picks: {self.menagerie_picks}
            Time elapsed M: {self.time_elapsed_meta_game_solution}
            Time elapsed Winrate matrix: {self.time_elapsed_meta_game_update}
            '''
            return s
