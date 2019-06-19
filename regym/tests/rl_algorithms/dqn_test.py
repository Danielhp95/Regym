from regym.rl_algorithms.agents import build_DQN_Agent
from regym.rl_algorithms import rockAgent

from test_fixtures import RPSTask, dqn_config_dict


def test_dqn_can_take_actions(RPSTask, dqn_config_dict):
    env = RPSTask.env
    agent = build_DQN_Agent(RPSTask, dqn_config_dict, 'DQN')
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = env.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        observation, rewards, done, info = env.step([a, a])
        # TODO technical debt
        # assert RPSenv.observation_space.contains([a, a])
        # assert RPSenv.action_space.contains([a, a])


def test_learns_to_beat_rock_in_RPS(RPSTask, dqn_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from rps_test import learns_against_fixed_opponent_RPS

    agent = build_DQN_Agent(RPSTask, dqn_config_dict, 'DQN')
    agent.training = True
    learns_against_fixed_opponent_RPS(agent, fixed_opponent=rockAgent,
                                      total_episodes=5000, training_percentage=0.95,
                                      reward_threshold=0.2)
