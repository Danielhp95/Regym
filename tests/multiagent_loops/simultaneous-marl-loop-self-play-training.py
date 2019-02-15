import os
import sys

import gym

sys.path.append(os.path.abspath('../..'))
from multiagent_loops import simultaneous_action_rl_loop
from training_schemes import SelfPlayTrainingScheme
import rl_algorithms

from unittest.mock import Mock


def test_naive_self_play_with_simulatenous_marl_loop():
    def rockPlayingMockAlgorithm():
        m = Mock(spec=rl_algorithms.agents.TabularQLearningAgent)
        m.take_action.return_value = 0 # Always return ROCK
        m.name = 'test_name_TQL'
        return m

    mockSelfPlay = Mock(SelfPlayTrainingScheme)
    mockSelfPlay.opponent_sampling_distribution.return_value = [rockPlayingMockAlgorithm()]

    mockEnv = Mock(gym.Env)
    mockEnv.step.return_value = (None, [None, None], True, None) # state, reward, done, info

    mockAlgorithm = Mock(spec=rl_algorithms.agents.TabularQLearningAgent)
    mockAlgorithm.clone.return_value = rockPlayingMockAlgorithm()
    mockAlgorithm.take_action.return_value = 0 # Always return ROCK

    target_episodes = 10
    opci = 1

    simultaneous_action_rl_loop.self_play_training(env=mockEnv, training_agent=rl_algorithms.AgentHook(mockAlgorithm),
                                          self_play_scheme=mockSelfPlay, 
                                          target_episodes=target_episodes, 
                                          results_path='./test',
                                          opci=opci)

    assert mockSelfPlay.curator.call_count == target_episodes
    assert mockSelfPlay.opponent_sampling_distribution.call_count == target_episodes
    assert mockAlgorithm.take_action.call_count == target_episodes
    assert mockEnv.reset.call_count == target_episodes
    assert mockEnv.step.call_count == target_episodes
