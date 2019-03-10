import os
import sys
sys.path.append(os.path.abspath('../../'))

from test_fixtures import ddpg_config_dict, RoboSumoenv, RoboSumoTask
from rl_algorithms.agents import build_DDPG_Agent


def test_ddpg_can_take_actions(RoboSumoenv, RoboSumoTask, ddpg_config_dict):
    agent = build_DDPG_Agent(RoboSumoTask, ddpg_config_dict, 'DDPG')
    RoboSumoenv.reset()
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = RoboSumoenv.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        print(i,a)
        observation, rewards, done, info = RoboSumoenv.step([a, a])
        # TODO technical debt
        # assert RoboSumoenv.observation_space.contains([a, a])
        # assert RoboSumoenv.action_space.contains([a, a])

"""
def test_learns_to_beat_rock_in_RPS(RPSTask, ppo_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from rps_test import learns_against_fixed_opponent_RPS

    agent = build_DDPG_Agent(RPSTask, ppo_config_dict, 'DDPG')
    assert agent.training
    learns_against_fixed_opponent_RPS(agent, fixed_opponent=rockAgent,
                                      total_episodes=1000, training_percentage=0.9,
                                      reward_threshold=0.1)
"""

if __name__ == "__main__":
    test_ddpg_can_take_actions(RoboSumoenv(), RoboSumoTask(RoboSumoenv()), ddpg_config_dict())