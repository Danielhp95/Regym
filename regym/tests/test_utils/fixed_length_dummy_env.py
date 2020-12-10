from typing import Any, List

import pytest
import gym
from gym.spaces import Discrete, Tuple


@pytest.fixture
def FixedLengthDummyTask():
    from regym.environments import generate_task, EnvType
    from gym.envs.registration import register
    register(id='FixedLengthDummy-v0', entry_point='regym.tests.test_utils.fixed_length_dummy_env:FixedLengthDummyEnv')
    return generate_task('FixedLengthDummy-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)


class FixedLengthDummyEnv(gym.Env):
    '''
    For testing purposes

    A _sequential_ multiagent environment that lasts for a given fixed number
    of episodes.  Suitable for arbitrary number of agents.
    Enviroment returns always 0 rewards and has no action or obser
    '''
    def __init__(self, episode_length: int = 3, num_players: int = 2,
                 final_rewards: List[float] = [0., 0.]):
        assert len(final_rewards) == num_players, ('If a list of final_rewards is '
                                                  'specified, it must be a of the '
                                                  'the same length as the number of players. '
                                                  f'Final rewards length: {len(final_rewards)} '
                                                  f'Num players: {num_players}')
        self.episode_length = episode_length
        self.current_step = 1

        self.num_players = num_players
        self.current_player = 0

        self.observation_space = Tuple([Discrete(1)
                                        for _ in range(self.num_players)])
        self.action_space = Tuple([Discrete(1)
                                   for _ in range(self.num_players)])

        self.final_rewards = final_rewards

    def reset(self):
        self.current_step = 0
        self.next_agent_turn = 0
        obs = [[0.] for _ in range (self.num_players)]
        return obs

    def step(self, action: Any):
        self.current_step += 1
        self.current_player = (self.current_player + 1) % self.num_players

        done = (self.current_step >= self.episode_length)
        if done:
            rewards = self.final_rewards
        else:
            rewards = [0] * self.num_players
        succ_obs = [[self.current_step] for _ in range(self.num_players)]
        info = {'current_player': self.current_player} if not done else {}

        return succ_obs, rewards, done, info
