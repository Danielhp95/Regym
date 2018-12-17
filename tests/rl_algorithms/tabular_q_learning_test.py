import os
import sys
sys.path.append(os.path.abspath('../../'))

from rl_algorithms.tabular_q_learning import TabularQLearning
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
    algorithm = TabularQLearning(env.state_space_size, env.action_space_size,
                                 env.hashing_function, learning_rate, training)
    assert algorithm.Q_table.shape == (env.state_space_size, env.action_space_size)
    assert algorithm.learning_rate == learning_rate
    assert algorithm.hashing_function == env.hashing_function
    assert algorithm.training == training


def test_cloning(env):
    algorithm = TabularQLearning(env.state_space_size, env.action_space_size, env.hashing_function)
    algorithm2 = algorithm.clone()
    assert algorithm != algorithm2 # Check that objects aren't pointing to same address
    assert algorithm.hashing_function == algorithm2.hashing_function # Check same hashing function is used
    assert np.array_equal(algorithm.Q_table, algorithm2.Q_table)
