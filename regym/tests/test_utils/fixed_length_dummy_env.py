from typing import Any

import gym
from gym.spaces import Discrete, Tuple


class FixedLengthDummyEnv(gym.Env):
    '''
    For testing purposes

    A _sequential_ multiagent environment that lasts for a given fixed number
    of episodes.  Suitable for arbitrary number of agents.
    Enviroment returns always 0 rewards and has no action or obser
    '''
    def __init__(self, episode_length: int, num_players: int = 2):
        self.episode_length = episode_length
        self.current_step = 0

        self.num_players = num_players
        self.current_player = 0

        self.observation_space = Tuple([Discrete(1)
                                        for _ in range(self.num_players)])
        self.action_space = Tuple([Discrete(1)
                                   for _ in range(self.num_players)])

    def reset(self):
        self.current_step = 0
        self.next_agent_turn = 0
        obs = [[] for _ in range (self.num_players)]
        return obs

    def step(self, action: Any):
        self.current_step += 1
        self.current_player = (self.current_player + 1) % self.num_players

        rewards = [0] * self.num_players
        succ_obs = [[] for _ in range(self.num_players)]
        done = (self.current_step >= self.episode_length)
        info = {'current_player': self.current_player} if not done else {}

        return succ_obs, rewards, done, info
