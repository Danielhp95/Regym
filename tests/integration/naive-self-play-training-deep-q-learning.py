import os
import sys
import torch

sys.path.append(os.path.abspath('../..'))
from multiagent_loops import simultaneous_action_rl_loop
from training_schemes import NaiveSelfPlay
from rl_algorithms import build_DQN_Agent
import numpy as np

import gym
import gym_rock_paper_scissors

def test_selfplay():
	env = gym.make('RockPaperScissors-v0')
	env.__init__(stacked_observations=2, max_repetitions=1)

	training_policy = build_DQN_Agent(state_space_size=env.state_space_size, action_space_size=env.action_space_size, hash_function=env.hash_state, double=True, dueling=True)

	policy = simultaneous_action_rl_loop.self_play_training(env=env, training_policy=training_policy,
	                                                        self_play_scheme=NaiveSelfPlay, target_episodes=10, opci=1)

	training_policy.stop_training()
	#assert np.array_equal(policy.Q_table, training_policy.Q_table)
	#np.savetxt('policy', policy.Q_table) # To manually inspect the policy

def test_selfplay_DQNvsFixedAgent():
	target_episodes = 1000
	env = gym.make('RockPaperScissors-v0')
	env.__init__(stacked_observations=2, max_repetitions=1)

	training_policy = build_DQN_Agent(state_space_size=env.state_space_size, 
										action_space_size=env.action_space_size, 
										hash_function=env.hash_state, 
										double=True, 
										dueling=True,
										num_worker=1,
										MIN_MEMORY=1e1,
										epsstart=0.5,
										epsend=0.3,
										epsdecay=1e2)

	fixedAgent = gym_rock_paper_scissors.fixed_agents.rockAgent

	training_policy.launch_training()
	
	policy = simultaneous_action_rl_loop.self_play_training(env=env,
															menagerie=[fixedAgent],
															training_policy=training_policy,
	                                                        self_play_scheme=NaiveSelfPlay, 
	                                                        target_episodes=target_episodes, 
	                                                        opci=1)

	training_policy.stop_training()
	#assert np.array_equal(policy.Q_table, training_policy.Q_table)
	#np.savetxt('policy', policy.Q_table) # To manually inspect the policy

if __name__ == '__main__':
	torch.multiprocessing.set_start_method('spawn') 
	#test_selfplay()
	test_selfplay_DQNvsFixedAgent()