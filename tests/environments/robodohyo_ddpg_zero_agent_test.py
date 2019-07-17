import os
import sys
sys.path.append(os.path.abspath('../../'))

from test_fixtures import  ddpg_config_dict_ma, RoboSumoenv, RoboSumoTask, RoboSumoWRSenv, RoboSumoWRSTask

from rl_algorithms.agents import build_DDPG_Agent
from rl_algorithms import AgentHook
from rl_algorithms.networks import PreprocessFunctionToTorch
from RoboSumo_test import learns_against_fixed_opponent_RoboSumo_parallel, record_against_fixed_opponent_RoboSumo

import numpy as np 

class RoboDohyoZeroAgent:
    def __init__(self, nbr_actor):
        self.nbr_actor = nbr_actor
        self.name = "ZeroAgent"
    def take_action(self, state):
        #return np.concatenate( [np.zeros((1,8), dtype="float32") for _ in range(self.nbr_actor)], axis=0)
        return np.concatenate( [np.asarray([[-0.5, 0.5, 0.25, -0.75, -0.75, -0.5, -0.5, -0.5]]) for _ in range(self.nbr_actor)], axis=0)
    def handle_experience(self, s, a, r, succ_s, done):
        pass 

def robodohyo_zero_agent(nbr_actor):
    return RoboDohyoZeroAgent(nbr_actor)

def ddpg_config_dict():
    config = dict()
    config['discount'] = 0.99
    config['tau'] = 1e-3
    config['use_cuda'] = True
    config['nbrTrainIteration'] = 1 
    config['action_scaler'] = 1.0 
    config['use_HER'] = False
    config['HER_k'] = 2
    config['HER_strategy'] = 'future'
    config['HER_use_singlegoal'] = False 
    config['use_PER'] = True 
    config['PER_alpha'] = 0.7 
    config['replay_capacity'] = 25e3
    config['min_capacity'] = 5e3 
    config['batch_size'] = 32#128
    config['learning_rate'] = 3.0e-4
    config['nbr_actor'] = 1#32
    return config

def RoboDohyoenv():
    import roboschool
    import gym
    #return gym.make('RoboschoolSumo-v0')
    return gym.make('RoboschoolSumoWithRewardShaping-v0')


def RoboDohyoTask(RoboSumoenv):
    from environments.gym_parser import parse_gym_environment
    return parse_gym_environment(RoboSumoenv)

def test_learns_to_beat_zero_in_RoboSumo(RoboSumoWRSTask, ddpg_config_dict_ma):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    load_agent = False
    
    if load_agent:
        agent = AgentHook.load(load_path='/tmp/test_DDPG_agent_RoboschoolSumoWithRewardShaping-v0.agent')
    else:
        agent = build_DDPG_Agent(RoboSumoWRSTask, ddpg_config_dict_ma, 'DDPG_agent')
    agent.training = True
    assert agent.training
    
    opponent = robodohyo_zero_agent(ddpg_config_dict_ma['nbr_actor'])
    
    envname = 'RoboschoolSumoWithRewardShaping-v0'
    learns_against_fixed_opponent_RoboSumo_parallel(agent, fixed_opponent=opponent,
                                      total_episodes=100, training_percentage=0.9,
                                      reward_threshold_percentage=0.25, envname=envname, nbr_parallel_env=ddpg_config_dict_ma['nbr_actor'], save=True)

def record_RoboDohyo_ZeroAgent(RoboDohyoTask, config_dict):
    load_agent = False
    
    if load_agent:
        agent = AgentHook.load(load_path='/tmp/test_ddpg_RoboschoolSumoWithRewardShaping-v0.agent')
    else:
        agent = build_DDPG_Agent(RoboDohyoTask, config_dict, 'DDPG_agent')
    agent.training = True
    assert agent.training
    
    opponent = robodohyo_zero_agent(config_dict['nbr_actor'])
    
    envname = 'RoboschoolSumoWithRewardShaping-v0'
    record_against_fixed_opponent_RoboSumo(agent, fixed_opponent=opponent, envname=envname)


if __name__ == "__main__":
    #test_learns_to_beat_rock_in_RoboSumo(RoboSumoTask(RoboSumoenv()), ddpg_config_dict_ma())
    record_RoboDohyo_ZeroAgent(RoboDohyoTask(RoboDohyoenv()), ddpg_config_dict())