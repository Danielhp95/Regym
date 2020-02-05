from typing import List, Tuple
from copy import copy
import numpy as np
import gym
from gym.spaces import Tuple, Discrete


class RandomWalkEnv(gym.Env):
    '''
    Simultaneous test environment for coordination and planning.
    Two agents move in a 1D line. If _any_ agent reaches a target position,
    that agent receives a reward and the episode is terminated.
    Because the environment is simultaneous, both agents can reach
    the target position at the same time, leading to a reward to both agents.

    Action space: {0: Moving right, 1: Moving left}
    Observation space (Fully observable): [position player 0 (int), position player 1 (int)]
    '''

    def __init__(self, target=3, starting_positions=[0, 0], space_size=50):
        self.action_space = Tuple([Discrete(2), Discrete(2)])
        self.starting_positions = starting_positions
        self.space_size = space_size
        self.observation_space = Tuple([Discrete(space_size), Discrete(space_size)])

        self.target = target
        self.done = False
        self.reset()

    def reset(self) -> List[np.ndarray]:
        '''
        Restores environment initia state.
        :returns: Environment initial state (starting positions for all agents)
        '''
        self.winner = -1
        self.done = False
        self.current_positions = self.starting_positions
        return [np.array(copy(self.current_positions)), np.array(copy(self.current_positions))]

    def clone(self):
        return RandomWalkEnv(target=self.target, starting_positions=copy(self.current_positions), space_size=self.space_size)

    def step(self, actions: List) -> Tuple:
        """
        :param actions: List of two elements, containing one action for each player
        :returns: OpenAI gym standard (succ_observations, reward_vector, done, info)
        """
        if self.done:
            return [self.current_positions[0]], [self.current_positions[1]], [0, 0], self.done, {}
        for i in range(2):
            if actions[i] == 0:
                self.current_positions[i] += 1
            elif actions[i] == 1:
                self.current_positions[i] -= 1
            else:
                pass

        reward_vector = [1 if self.target == p else 0 for p in self.current_positions]

        if reward_vector[1] == 1:
            self.winner = 1
        elif reward_vector[0] == 1:
            self.winner = 0
        else:
            self.winner = -1

        info = {}
        self.done = self.winner != -1
        return [np.array(copy(self.current_positions)), np.array(copy(self.current_positions))], reward_vector, self.done, info

    def is_over(self) -> bool:
        '''
        Whether a player has reached the target position
        :returns: flag of episode termination
        '''
        return self.winner != -1

    def get_moves(self, player_id: int) -> List[int]:
        """
        :returns: array with all possible moves, index of columns which aren't full
        """
        if self.winner != -1:
            return []
        return [0, 1]

    def get_result(self, player_id) -> int:
        """
        :param player: (int) player which we want to see if he / she is a winner
        :returns: winner from the perspective of the param player
        """
        return int(self.current_positions[player_id] == self.target)

    def render(self, mode='human') -> str:
        return f'Current position: {self.current_positions}. Target: {self.target}'
