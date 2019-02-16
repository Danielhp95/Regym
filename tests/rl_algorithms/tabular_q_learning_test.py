import os
import sys
sys.path.append(os.path.abspath('../../'))

from rl_algorithms.TQL import TabularQLearningAlgorithm
import numpy as np
import pytest


@pytest.fixture
def env():
    class Env():
        state_space_size  = 5
        action_space_size = 5
        hashing_function = lambda x: x
    return Env()


def test_algorithm_instantiation(env):
    learning_rate = 0.1
    training = False
    algorithm = TabularQLearningAlgorithm(env.state_space_size, env.action_space_size,
                                 env.hashing_function, learning_rate, training)
    assert algorithm.Q_table.shape == (env.state_space_size, env.action_space_size)
    assert algorithm.learning_rate == learning_rate
    assert algorithm.hashing_function == env.hashing_function
    assert algorithm.training == training
