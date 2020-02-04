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
        self.winner = -1
        self.done = False
        self.current_positions = self.starting_positions

    def reset(self):
        self.winner = -1
        self.done = False
        self.current_positions = self.starting_positions
        return [np.array(copy(self.current_positions)), np.array(copy(self.current_positions))]

    def clone(self):
        return RandomWalkEnv(target=self.target, starting_positions=copy(self.current_positions), space_size=self.space_size)

    def step(self, actions: List):
        """
        :param actions: List of two elements, containing one action for each player
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

        # TODO: find if a player has won
        if reward_vector[1] == 1:
            self.winner = 1
        elif reward_vector[0] == 1:
            self.winner = 0
        else:
            self.winner = -1

        # info should be kept empty
        info = {}
        self.done = self.winner != -1
        return [np.array(copy(self.current_positions)), np.array(copy(self.current_positions))], reward_vector, self.done, info

    def is_over(self):
        return self.winner != -1

    def get_moves(self, player_id: int):
        """
        :returns: array with all possible moves, index of columns which aren't full
        TODO: figure out what are the valid moves an agent can take.
        (i.e figure ability cooldowns / collision against map borders)
        """
        if self.winner != -1:
            return []
        return [0, 1] # Left, right, null

    def get_result(self, player_id):
        """
        :param player: (int) player which we want to see if he / she is a winner
        :returns: winner from the perspective of the param player
        """
        return int(self.current_positions[player_id] == self.target)

    def render(self, mode='human'):
        return f'Current position: {self.current_positions}. Target: {self.target}'
