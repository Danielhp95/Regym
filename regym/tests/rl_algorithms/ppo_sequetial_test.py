from test_fixtures import ppo_config_dict, ppo_rnn_config_dict, KuhnTask

from regym.rl_algorithms.agents import build_PPO_Agent
from regym.rl_algorithms import rockAgent


def test_ppo_can_take_actions(KuhnTask, ppo_config_dict):
    agent = build_PPO_Agent(KuhnTask, ppo_config_dict, 'PPO')
    act_in_task_env(KuhnTask, agent)


def test_ppo_rnn_can_take_actions(KuhnTask, ppo_rnn_config_dict):
    agent = build_PPO_Agent(KuhnTask, ppo_rnn_config_dict, 'RNN_PPO')
    act_in_task_env(KuhnTask, agent)


def act_in_task_env(task, agent):
    done = False
    env = task.env
    env.reset()
    while not done:
        # asumming that first observation corresponds to observation space of this agent
        random_observation = env.observation_space.sample()[0]
        a = agent.take_action(random_observation)
        observation, rewards, done, info = env.step(a)


def test_learns_to_beat_rock_in_kuhn_poker(KuhnTask, ppo_config_dict):
    build_agent_func = lambda: build_PPO_Agent(KuhnTask, ppo_config_dict, 'PPO-MLP')
    play_kuhn_poker_all_positions_all_fixed_agents(build_agent_func)


def test_learns_to_beat_rock_in_kuhn_poker_rnn(KuhnTask, ppo_rnn_config_dict):
    build_agent_func = lambda: build_PPO_Agent(KuhnTask, ppo_rnn_config_dict, 'PPO-RNN')
    play_kuhn_poker_all_positions_all_fixed_agents(build_agent_func)


def play_kuhn_poker_all_positions_all_fixed_agents(build_agent_func):
    # agent = build_agent_func()
    # play_against_fixed_agent(agent, fixed_agent_action=0, agent_position=0,
    #                          max_reward=1.)
    # agent = build_agent_func()
    # play_against_fixed_agent(agent, fixed_agent_action=1, agent_position=0,
    #                          max_reward=0.5, total_episodes=80000)
    # agent = build_agent_func()
    # play_against_fixed_agent(agent, fixed_agent_action=0, agent_position=1,
    #                          max_reward=1.)
    agent = build_agent_func()
    play_against_fixed_agent(agent, fixed_agent_action=1, agent_position=1,
                             max_reward=0.5, total_episodes=10000)


def play_against_fixed_agent(agent, fixed_agent_action, agent_position,
                             max_reward, total_episodes=2000):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from play_against_fixed_opponent import learn_against_fix_opponent

    class FixedAgent():
        def __init__(self, action):
            self.name = f'action: {action}'
            self.action = action

        def take_action(self, *args):
            return self.action

        def handle_experience(self, *args):
            pass

    fixed_opponent = FixedAgent(fixed_agent_action)
    assert agent.training
    learn_against_fix_opponent(agent, fixed_opponent=fixed_opponent,
                               agent_position=agent_position,
                               env='KuhnPoker-v0', env_type='sequential',
                               total_episodes=total_episodes, training_percentage=0.9,
                               reward_threshold=0.1,
                               maximum_average_reward=max_reward,
                               evaluation_method='last')