from test_fixtures import ppo_config_dict, ppo_rnn_config_dict, RPSTask

from regym.rl_algorithms.agents import build_PPO_Agent
from regym.rl_algorithms import rockAgent


def test_ppo_can_take_actions(RPSTask, ppo_config_dict):
    env = RPSTask.env
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = env.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        observation, rewards, done, info = env.step([a, a])
        # TODO technical debt
        # assert RPSenv.observation_space.contains([a, a])
        # assert RPSenv.action_space.contains([a, a])


def test_learns_to_beat_rock_in_RPS(RPSTask, ppo_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from play_against_fixed_opponent import learn_against_fix_opponent

    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    assert agent.training
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0, # Doesn't matter in RPS
                               env=RPSTask.name, env_type='simultaneous',
                               total_episodes=500, training_percentage=0.9,
                               reward_threshold=0.1,
                               maximum_average_reward=1.0,
                               evaluation_method='average')


def test_ppo_rnn_can_take_actions(RPSTask, ppo_rnn_config_dict):
    env = RPSTask.env
    agent = build_PPO_Agent(RPSTask, ppo_rnn_config_dict, 'RNN_PPO')
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = env.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        observation, rewards, done, info = env.step([a, a])
        # TODO technical debt
        # assert RPSenv.observation_space.contains([a, a])
        # assert RPSenv.action_space.contains([a, a])


def test_learns_to_beat_rock_in_RPS_rnn(RPSTask, ppo_rnn_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from play_against_fixed_opponent import learn_against_fix_opponent

    agent = build_PPO_Agent(RPSTask, ppo_rnn_config_dict, 'RNN_PPO')
    assert agent.training
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0, # Doesn't matter in RPS
                               env=RPSTask.name, env_type='simultaneous',
                               total_episodes=500, training_percentage=0.9,
                               reward_threshold=0.1,
                               maximum_average_reward=1.0,
                               evaluation_method='average')
