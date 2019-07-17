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

if __name__ == "__main__":
    test_ddpg_can_take_actions(RoboSumoenv(), RoboSumoTask(RoboSumoenv()), ddpg_config_dict())