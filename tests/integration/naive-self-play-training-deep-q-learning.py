import os
import sys
import torch

sys.path.append(os.path.abspath('../..'))
from multiagent_loops import simultaneous_action_rl_loop
from training_schemes import NaiveSelfPlay
from rl_algorithms import build_DQN_Agent, AgentHook
import numpy as np

import gym
import gym_rock_paper_scissors
from gym_rock_paper_scissors.fixed_agents import rockAgent, paperAgent, scissorsAgent
import environments


def test_selfplay_DQNvsFixedAgent():
	torch.multiprocessing.set_start_method('spawn')
	target_episodes = 1000
	env = gym.make('RockPaperScissors-v0')
	env.__init__(stacked_observations=2, max_repetitions=1)
	task = environments.parse_gym_environment(env)

	training_policy = build_DQN_Agent(state_space_size=task.observation_dim,
                                          action_space_size=task.action_dim,
                                          double=True,
                                          dueling=True,
                                          num_worker=1,
                                          MIN_MEMORY=1e1,
                                          epsstart=0.9,
                                          epsend=0.01,
                                          epsdecay=2e2,
                                          use_cuda=False)

	fixedAgent = rockAgent

	policy = simultaneous_action_rl_loop.self_play_training(env=env,
                                                                menagerie=[fixedAgent],
                                                                training_agent=AgentHook(training_policy),
	                                                        self_play_scheme=NaiveSelfPlay,
	                                                        target_episodes=target_episodes,
	                                                        results_path='/tmp',
	                                                        opci=1)


if __name__ == '__main__':
	test_selfplay_DQNvsFixedAgent()
