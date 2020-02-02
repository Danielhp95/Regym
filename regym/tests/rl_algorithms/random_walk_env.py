from typing import List
from copy import copy
import numpy as np
import gym
from gym.spaces import Tuple, Discrete


class RandomWalkEnv(gym.Env):

    def __init__(self, target=3, starting_positions=[0, 0], space_size=50):
        self.action_space = Tuple([Discrete(2), Discrete(2)]) # Left, Right, Null
        self.starting_positions = starting_positions
        self.space_size = space_size
        self.observation_space = Tuple([Discrete(space_size), Discrete(space_size)])

        self.target = target
        self.done = False
        self.reset()

    def reset(self):
        self.winner = -1
        self.done = False
        self.current_positions = self.starting_positions
        return np.array([copy(self.current_positions), copy(self.current_positions)])

    def clone(self):
        return RandomWalkEnv(target=self.target, starting_positions=copy(self.current_positions), space_size=self.space_size)

    def step(self, actions: List):
        """
        :param actions: List of two elements, containing one action for each player
        """
        # TODO: in here we can enforce collaboration (i.e They only move if both players say right)
        for i, a in enumerate(actions):
            if a == 0:
                self.current_positions[i] += 1
            elif a == 1:
                self.current_positions[i] -= 1

        reward_vector = [int(self.target == p) for p in self.current_positions]

        if reward_vector[0] == 1:
            self.winner = 1
        elif reward_vector[1] == 1:
            self.winner = 2

        # info should be kept empty
        info = {}
        self.done = self.winner != -1
        return [np.array([copy(self.current_positions), copy(self.current_positions)])], reward_vector, self.done, info

    def get_moves(self, player_id: int):
        """
        :returns: array with all possible moves, index of columns which aren't full
        TODO: figure out what are the valid moves an agent can take.
        (i.e figure ability cooldowns / collision against map borders)
        """
        if self.winner != 0:
            return []
        return [0, 1] # Left, right, null

    def get_result(self, player):
        """
        :param player: (int) player which we want to see if he / she is a winner
        :returns: winner from the perspective of the param player
        """
        return player == self.winner

    def render(self, mode='human'):
        return f'Current position: {self.current_positions}. Target: {self.target}'
