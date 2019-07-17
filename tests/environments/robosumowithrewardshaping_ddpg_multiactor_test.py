import os
import sys
sys.path.append(os.path.abspath('../../'))

from test_fixtures import ddpg_config_dict_ma, RoboSumoenv, RoboSumoTask

from rl_algorithms.agents import build_DDPG_Agent
from rl_algorithms import AgentHook
from RoboSumo_test import learns_against_fixed_opponent_RoboSumo_parallel, record_against_fixed_opponent_RoboSumo


'''
def test_ddpg_can_take_actions(RoboSumoenv, RoboSumoTask, ddpg_config_dict_ma):
    agent = build_DDPG_Agent(RoboSumoTask, ddpg_config_dict_ma, 'DDPG')
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

def test_learns_to_beat_rock_in_RoboSumo(RoboSumoTask, ddpg_config_dict_ma):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    load_agent = False
    load_opponent = False
    
    if load_agent:
        agent = AgentHook.load(load_path='/tmp/test_ddpg_roboschoolsumo_multiactor-1.agent')
    else:
        agent = build_DDPG_Agent(RoboSumoTask, ddpg_config_dict_ma, 'DDPG_agent')
    agent.training = True
    assert agent.training
    
    if load_opponent:
        opponent = AgentHook.load(load_path='/tmp/test_ddpg_roboschoolsumo_multiactor-1.agent')
    else:
        opponent = build_DDPG_Agent(RoboSumoTask, ddpg_config_dict_ma, 'DDPG_opp')
    opponent.training = False

    envname = 'RoboschoolSumoWithRewardShaping-v0'
    learns_against_fixed_opponent_RoboSumo_parallel(agent, fixed_opponent=opponent,
                                      total_episodes=100, training_percentage=0.9,
                                      reward_threshold_percentage=0.25, envname=envname, nbr_parallel_env=ddpg_config_dict_ma['nbr_actor'], save=True)


def record_RoboSumo(RoboSumoTask, ddpg_config_dict_ma):
    load_agent = True
    load_opponent = False
    
    if load_agent:
        agent = AgentHook.load(load_path='/tmp/test_DDPG_agent_RoboschoolSumoWithRewardShaping-v0.agent')#/tmp/test_ddpg_RoboschoolSumoWithRewardShaping-v0.agent')
    else:
        agent = build_DDPG_Agent(RoboSumoTask, ddpg_config_dict_ma, 'DDPG_agent')
    agent.training = True
    assert agent.training
    
    if load_opponent:
        opponent = AgentHook.load(load_path='/tmp/test_ddpg_RoboschoolSumoWithRewardShaping-v0.agent')
    else:
        opponent = build_DDPG_Agent(RoboSumoTask, ddpg_config_dict_ma, 'DDPG_opp')
    opponent.training = False

    envname = 'RoboschoolSumoWithRewardShaping-v0'
    record_against_fixed_opponent_RoboSumo(agent, fixed_opponent=opponent, envname=envname)


if __name__ == "__main__":
    #test_learns_to_beat_rock_in_RoboSumo(RoboSumoTask(RoboSumoenv()), ddpg_config_dict_ma())
    record_RoboSumo(RoboSumoTask(RoboSumoenv()), ddpg_config_dict_ma())