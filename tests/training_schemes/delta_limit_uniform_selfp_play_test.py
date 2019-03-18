import os
import sys
sys.path.append(os.path.abspath('../../'))

import copy
import gym
from tqdm import tqdm

from test_fixtures import ppo_config_dict, RPSenv, RPSTask

from rl_algorithms.agents import build_PPO_Agent
from rl_algorithms import AgentHook

import util
from training_schemes import DeltaLimitUniformSelfPlay

from training_schemes_test import self_play_training

def test_delta_limit_uniform_selfplay(RPSenv, RPSTask, ppo_config_dict):
    training_agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO_agent')
    opponent = build_DDPG_Agent(RPSTask, ppo_config_dict, 'PPO_opp')

    menagerie = [opponent]
    opci = 10
    target_episodes = 1000
    men, agent, tr = self_play_training(env=RPSenv, training_agent=training_agent, self_play_scheme=DeltaLimitUniformSelfPlay, target_episodes=target_episodes, opci=opci, menagerie=menagerie, menagerie_path=None, iteration=None)

if __name__ == "__main__":
    test_delta_limit_uniform_selfplay(RPSenv(), RPSTask(RPSenv()), ppo_config_dict())