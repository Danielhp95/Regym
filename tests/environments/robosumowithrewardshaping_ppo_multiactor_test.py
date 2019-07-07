import os
import sys
sys.path.append(os.path.abspath('../../'))

from test_fixtures import ppo_config_dict, ppo_config_dict_ma, RoboSumoenv, RoboSumoTask, RoboSumoWRSenv, RoboSumoWRSTask

from regym.rl_algorithms.agents import build_PPO_Agent
from regym.rl_algorithms import AgentHook
from RoboSumo_test import learns_against_fixed_opponent_RoboSumo_parallel, record_against_fixed_opponent_RoboSumo

import numpy as np 

'''
def test_ppo_can_take_actions(RoboSumoenv, RoboSumoTask, ppo_config_dict_ma):
    agent = build_PPO_Agent(RoboSumoTask, ppo_config_dict_ma)
    RoboSumoenv.reset()
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = RoboSumoenv.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        observation, rewards, done, info = RoboSumoenv.step([a, a])
        # TODO technical debt
        # assert RoboSumoenv.observation_space.contains([a, a])
        # assert RoboSumoenv.action_space.contains([a, a])
'''

"""
def test_learns_to_beat_rock_in_RoboSumo(RoboSumoTask, ppo_config_dict_ma):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    load_agent = False
    load_opponent = False
    
    if load_agent:
        agent = AgentHook.load(load_path='/tmp/test_ppo_roboschoolsumo_multiactor-1.agent')
    else:
        agent = build_PPO_Agent(RoboSumoTask, ppo_config_dict_ma, 'PPO_agent')
    agent.training = True
    assert agent.training
    
    if load_opponent:
        opponent = AgentHook.load(load_path='/tmp/test_ppo_roboschoolsumo_multiactor-1.agent')
    else:
        opponent = build_PPO_Agent(RoboSumoTask, ppo_config_dict_ma, 'PPO_opp')
    opponent.training = False

    envname = 'RoboschoolSumoWithRewardShaping-v0'
    learns_against_fixed_opponent_RoboSumo_parallel(agent, fixed_opponent=opponent,
                                      total_episodes=1000, training_percentage=0.9,
                                      reward_threshold_percentage=0.25, envname=envname, nbr_parallel_env=ppo_config_dict_ma['nbr_actor'], save=True)

"""


class RoboDohyoZeroAgent:
    def __init__(self, nbr_actor):
        self.nbr_actor = nbr_actor
        self.name = "ZeroAgent"
    def take_action(self, state):
        #return np.concatenate( [np.zeros((1,8), dtype="float32") for _ in range(self.nbr_actor)], axis=0)
        return np.concatenate( [np.asarray([[-0.5, 0.5, 0.5, -0.75, -0.5, -0.5, -0.5, -0.5]]) for _ in range(self.nbr_actor)], axis=0)
    def handle_experience(self, s, a, r, succ_s, done):
        pass 

    def set_nbr_actor(self, nbr_actor):
        pass

    def reset_actors(self):
        pass 

    def update_actors(self, actor_idx):
        pass 




def robodohyo_zero_agent(nbr_actor):
    return RoboDohyoZeroAgent(nbr_actor)

def ppo_config_dict():
    config = dict()
    config['discount'] = 0.995
    config['use_gae'] = True
    config['use_cuda'] = True
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 20
    config['mini_batch_size'] = 256
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-3
    config['adam_eps'] = 1.0e-5
    config['nbr_actor'] = 1
    config['horizon'] = 8192
    return config

def RoboDohyoenv():
    import roboschool
    import gym
    return gym.make('RoboschoolSumoWithRewardShaping-v0')
    #return gym.make('RoboschoolSumo-v0')


def RoboDohyoTask(RoboSumoenv):
    from environments.gym_parser import parse_gym_environment
    return parse_gym_environment(RoboSumoenv)

def test_learns_to_beat_zero_in_RoboSumo(RoboSumoWRSTask, ppo_config_dict_ma):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    load_agent = False
    
    if load_agent:
        agent = AgentHook.load(load_path='/tmp/test_PPO_agent_RoboschoolSumoWithRewardShaping-v0.agent')
    else:
        agent = build_PPO_Agent(RoboSumoWRSTask, ppo_config_dict_ma, 'PPO_agent')
    agent.training = True
    assert agent.training
    
    opponent = robodohyo_zero_agent(ppo_config_dict_ma['nbr_actor'])
    
    envname = 'RoboschoolSumoWithRewardShaping-v0'
    learns_against_fixed_opponent_RoboSumo_parallel(agent, fixed_opponent=opponent,
                                      total_episodes=25, training_percentage=0.9,
                                      reward_threshold_percentage=0.25, envname=envname, nbr_parallel_env=ppo_config_dict_ma['nbr_actor'], save=True)

def record_RoboDohyo_ZeroAgent(RoboDohyoTask, ppo_config_dict):
    load_agent = False
    
    if load_agent:
        agent = AgentHook.load(load_path='/tmp/test_ppo_RoboschoolSumoWithRewardShaping-v0.agent')
    else:
        agent = build_PPO_Agent(RoboDohyoTask, ppo_config_dict, 'PPO_agent')
    agent.training = True
    assert agent.training
    
    opponent = robodohyo_zero_agent(ppo_config_dict['nbr_actor'])
    
    envname = 'RoboschoolSumoWithRewardShaping-v0'
    record_against_fixed_opponent_RoboSumo(agent, fixed_opponent=opponent, envname=envname)


if __name__ == "__main__":
    #test_learns_to_beat_zero_in_RoboSumo(RoboSumoTask(RoboSumoenv()), ppo_config_dict_ma())
    record_RoboDohyo_ZeroAgent(RoboDohyoTask(RoboDohyoenv()), ppo_config_dict())