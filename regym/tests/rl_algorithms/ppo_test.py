from functools import reduce
from test_fixtures import ppo_config_dict, ppo_rnn_config_dict, RPSTask, KuhnTask, CartPoleTask

import tqdm

import regym
from regym.tests.test_utils.play_against_fixed_opponent import learn_against_fix_opponent
from regym.networks.preprocessing import batch_vector_observation, flatten_and_turn_into_single_element_batch
from regym.environments import generate_task, EnvType
from regym.rl_algorithms.agents import Agent, build_Deterministic_Agent
from regym.rl_algorithms.agents import build_PPO_Agent
from regym.rl_algorithms import rockAgent


def test_ppo_can_take_actions(RPSTask, ppo_config_dict):
    env = RPSTask.env
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    number_of_actions = 30
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = env.observation_space.sample()[0]
        a = agent.model_free_take_action(random_observation)
        observation, rewards, done, info = env.step([a, a])
        # TODO technical debt
        # assert RPSenv.observation_space.contains([a, a])
        # assert RPSenv.action_space.contains([a, a])


def test_ppo_can_solve_multi_env_cartpole(CartPoleTask, ppo_config_dict):
    agent = build_PPO_Agent(CartPoleTask, ppo_config_dict, 'PPO-CartPole-Test')
    agent.state_preprocessing = batch_vector_observation  # Required for multiactor

    from torch.utils.tensorboard import SummaryWriter
    regym.rl_algorithms.PPO.ppo_loss.summary_writer = SummaryWriter('ppo_test_tensorboard')

    test_trajectories = multiactor_task_test(CartPoleTask, agent, train_episodes=5000, test_episodes=100)
    #test_trajectories = singleactor_task_test(CartPoleTask, agent, train_episodes=5000, test_episodes=100)

    max_traj_len = 200
    solved_threshold = 100
    total_test_trajectory_len = reduce(lambda acc, t: acc + len(t),
                                       test_trajectories, 0)
    assert total_test_trajectory_len / len(test_trajectories) >= solved_threshold


from regym.util import profile
@profile(filename='singleactor_task_profile.pstats')
def singleactor_task_test(task, agent, train_episodes: int, test_episodes: int):
    agent.training = True
    agent.algorithm.horizon = 2046
    agent.algorithm.mini_batch_size = 256
    assert agent.training, 'Agent should be training in order to solve test environment'
    import tqdm
    progress_bar = tqdm.tqdm(range(train_episodes))
    for _ in progress_bar:
        trajectory = task.run_episode([agent], training=True)
        progress_bar.set_description(f'{agent.name} in {task.env.spec.id}. Episode length: {len(trajectory)}')
    agent.training = False
    test_trajectories = []
    progress_bar = tqdm.tqdm(range(test_episodes))
    for _ in progress_bar:
        trajectory = task.run_episode([agent], training=False)
        test_trajectories.append(trajectory)
        progress_bar.set_description(f'{agent.name} in {task.env.spec.id}. Episode length: {len(trajectory)}')
    return test_trajectories


@profile(filename='multiactor_task_profile.pstats')
def multiactor_task_test(task, agent, train_episodes, test_episodes):
    agent.training = True
    agent.algorithm.horizon = 2046
    agent.algorithm.mini_batch_size = 256
    assert agent.training, 'Agent should be training in order to solve test environment'
    train_trajectories = task.run_episodes([agent], training=True,
            num_episodes=train_episodes, num_envs=12)
    agent.training = False
    test_trajectories = task.run_episodes([agent], training=False,
            num_episodes=test_episodes, num_envs=12)
    return test_trajectories


def test_learns_to_beat_rock_in_RPS(RPSTask, ppo_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''

    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    agent.state_preprocessing = flatten_and_turn_into_single_element_batch
    assert agent.training
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0, # Doesn't matter in RPS
                               task=RPSTask,
                               training_episodes=250,
                               benchmark_every_n_episodes=0,
                               test_episodes=50,
                               reward_tolerance=1.,
                               maximum_average_reward=10.0,
                               evaluation_method='cumulative')


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


# TODO: refactor with function below which does the same thing!
def test_learns_to_beat_rock_in_RPS_rnn(RPSTask, ppo_rnn_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    agent = build_PPO_Agent(RPSTask, ppo_rnn_config_dict, 'RNN_PPO')
    assert agent.training
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0, # Doesn't matter in RPS
                               task=RPSTask,
                               training_episodes=250,
                               benchmark_every_n_episodes=0,
                               test_episodes=50,
                               reward_tolerance=1.,
                               maximum_average_reward=10.0,
                               evaluation_method='cumulative')


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
        a = agent.model_free_take_action(random_observation)
        observation, rewards, done, info = env.step(a)


#def test_mlp_architecture_learns_to_beat_kuhn_poker(KuhnTask, ppo_config_dict):
#    build_agent_func = lambda: build_PPO_Agent(KuhnTask, ppo_config_dict, 'PPO-MLP')
#    play_kuhn_poker_all_positions_all_fixed_agents(build_agent_func)
#
#
#def test_rnn_architecture_learns_to_beat_kuhn_poker_rnn(KuhnTask, ppo_rnn_config_dict):
#    build_agent_func = lambda: build_PPO_Agent(KuhnTask, ppo_rnn_config_dict, 'PPO-RNN')
#    play_kuhn_poker_all_positions_all_fixed_agents(build_agent_func)


def play_kuhn_poker_all_positions_all_fixed_agents(build_agent_func):
    agent = build_agent_func()
    play_against_fixed_agent(agent, fixed_agent_action=0, agent_position=0,
                             max_reward=1., total_episodes=1500)
    agent = build_agent_func()
    play_against_fixed_agent(agent, fixed_agent_action=1, agent_position=0,
                             max_reward=0.2, total_episodes=4500)
                             # TODO: Properly calculate max reward in this context
    agent = build_agent_func()
    play_against_fixed_agent(agent, fixed_agent_action=0, agent_position=1,
                             max_reward=1., total_episodes=1500)
    agent = build_agent_func()
    play_against_fixed_agent(agent, fixed_agent_action=1, agent_position=1,
                             max_reward=0.2, total_episodes=4500)
                             # TODO: Properly calculate max reward in this context


def play_against_fixed_agent(agent, fixed_agent_action, agent_position,
                             max_reward, total_episodes=2000):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    kuhn_task = generate_task('KuhnPoker-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)
    fixed_opponent = build_Deterministic_Agent(kuhn_task, {'action': fixed_agent_action})
    assert agent.training
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0, # Doesn't matter in RPS
                               task=RPSTask,
                               training_episodes=250,
                               benchmark_every_n_episodes=0,
                               test_episodes=50,
                               reward_tolerance=1.,
                               maximum_average_reward=10.0,
                               evaluation_method='cumulative')
