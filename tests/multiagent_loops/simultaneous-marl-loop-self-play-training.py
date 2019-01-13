import os
import sys

import gym

sys.path.append(os.path.abspath('../..'))
from multiagent_loops import multiagent_rl_loop
from training_schemes import SelfPlayTrainingScheme
import rl_algorithms

from unittest.mock import Mock


def test_naive_self_play_with_simulatenous_marl_loop():
    def rockPlayingMockAlgorithm():
        m = Mock(spec=rl_algorithms.TabularQLearning)
        m.act.return_value = 0 # always return rock
        return m

    mockselfplay = mock(selfplaytrainingscheme)
    mockselfplay.opponent_sampling_distribution.return_value = [rockplayingmockalgorithm()]

    mockenv = mock(gym.env)
    mockenv.step.return_value = (none, [none, none], true, none) # state, reward, done, info

    mockalgorithm = mock(spec=rl_algorithms.tabularqlearning)
    mockalgorithm.clone.return_value = rockplayingmockalgorithm()
    mockalgorithm.act.return_value = 0 # always return rock

    target_episodes = 10
    opci = 1

    multiagent_rl_loop.self_play_training(env=mockenv, training_policy=mockalgorithm,
                                          self_play_scheme=mockselfplay, target_episodes=target_episodes, opci=opci)

    assert mockselfplay.curator.call_count == target_episodes
    assert mockselfplay.opponent_sampling_distribution.call_count == target_episodes
    assert mockalgorithm.act.call_count == target_episodes
    assert mockEnv.reset.call_count == target_episodes
    assert mockEnv.step.call_count == target_episodes
