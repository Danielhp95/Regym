from test_fixtures import reinforce_config_dict, CartPoleTask

from regym.rl_algorithms.agents import build_Reinforce_Agent
from regym.rl_loops.singleagent_loops.rl_loop import run_episode


def test_reinforce_can_take_actions(CartPoleTask, reinforce_config_dict):
    env = CartPoleTask.env
    agent = build_Reinforce_Agent(CartPoleTask, reinforce_config_dict, 'Test-Reinforce')
    number_of_actions = 30
    for i in range(number_of_actions):
        random_observation = env.observation_space.sample()
        action = agent.take_action(random_observation)
        assert env.action_space.contains(action)
