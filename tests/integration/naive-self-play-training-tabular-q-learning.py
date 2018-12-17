import os
import sys

sys.path.append(os.path.abspath('../..'))
from multiagent_loops import simultaneous_action_rl_loop
from training_schemes import NaiveSelfPlay
from rl_algorithms import TabularQLearning
import numpy as np

import gym
import gym_rock_paper_scissors

env = gym.make('RockPaperScissors-v0')
env.__init__(stacked_observations=2, max_repetitions=1)

training_policy = TabularQLearning(env.state_space_size, env.action_space_size, env.hash_state)

policy = simultaneous_action_rl_loop.self_play_training(env=env, training_policy=training_policy,
                                                        self_play_scheme=NaiveSelfPlay, target_episodes=1, opci=1)

assert np.array_equal(policy.Q_table, training_policy.Q_table)
np.savetxt('policy', policy.Q_table) # To manually inspect the policy
