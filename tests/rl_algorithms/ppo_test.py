import os
import sys
sys.path.append(os.path.abspath('../../'))

from test_fixtures import ppo_config_dict, RPSenv, RPSTask

from rl_algorithms.agents import build_PPO_Agent
from rl_algorithms import rockAgent


def test_ppo_can_take_actions(RPSenv, RPSTask, ppo_config_dict):
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = RPSenv.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        observation, rewards, done, info = RPSenv.step([a, a])
        # TODO technical debt
        # assert RPSenv.observation_space.contains([a, a])
        # assert RPSenv.action_space.contains([a, a])


def test_learns_to_beat_rock_in_RPS(RPSTask, ppo_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    #from rps_test import learns_against_fixed_opponent_RPS_parallel
    from rps_test import learns_against_fixed_opponent_RPS

    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    assert agent.training
    learns_against_fixed_opponent_RPS(agent, fixed_opponent=rockAgent,
                                      total_episodes=1000, training_percentage=0.9,
                                      reward_threshold=0.1)
    '''
    learns_against_fixed_opponent_RPS_parallel(agent, fixed_opponent=rockAgent,
                                      total_episodes=2000, training_percentage=0.9,
                                      reward_threshold_percentage=0.1,nbr_parallel_env=agent.nbr_actor)
    '''