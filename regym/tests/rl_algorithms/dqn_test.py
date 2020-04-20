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
        a = agent.model_free_take_action(random_observation, legal_actions=[0, 1, 2])
        observation, rewards, done, info = env.step([a, a])
        assert RPSTask.env.action_space.contains([a, a])


def test_vanilla_DQN_learns_to_beat_rock_in_RPS(RPSTask, dqn_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from play_against_fixed_opponent import learn_against_fix_opponent

    from torch.utils.tensorboard import SummaryWriter
    import regym
    regym.rl_algorithms.DQN.dqn_loss.summary_writer = SummaryWriter('test_tensorboard')
    agent = build_DQN_Agent(RPSTask, dqn_config_dict, 'DQN')
    assert agent.training
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0,  # Doesn't matter in RPS
                               task=RPSTask,
                               total_episodes=250, training_percentage=0.9,
                               reward_tolerance=2.,
                               maximum_average_reward=10.0,
                               evaluation_method='cumulative')


def test_double_DQN_learns_to_beat_rock_in_RPS(RPSTask, dqn_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from play_against_fixed_opponent import learn_against_fix_opponent

    from torch.utils.tensorboard import SummaryWriter
    import regym
    regym.rl_algorithms.DQN.dqn_loss.summary_writer = SummaryWriter('test_tensorboard')
    dqn_config_dict['double'] = True
    agent = build_DQN_Agent(RPSTask, dqn_config_dict, 'Double_DQN')
    assert agent.training and agent.algorithm.use_double
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0,  # Doesn't matter in RPS
                               task=RPSTask,
                               total_episodes=250, training_percentage=0.9,
                               reward_tolerance=2,
                               maximum_average_reward=10.0,
                               evaluation_method='cumulative')


def test_dueling_DQN_learns_to_beat_rock_in_RPS(RPSTask, dqn_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from play_against_fixed_opponent import learn_against_fix_opponent

    from torch.utils.tensorboard import SummaryWriter
    import regym
    regym.rl_algorithms.DQN.dqn_loss.summary_writer = SummaryWriter('test_tensorboard')
    dqn_config_dict['dueling'] = True
    agent = build_DQN_Agent(RPSTask, dqn_config_dict, 'Dueling_DQN')
    assert agent.training and agent.algorithm.use_dueling
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0,  # Doesn't matter in RPS
                               task=RPSTask,
                               total_episodes=250, training_percentage=0.9,
                               reward_tolerance=2.,
                               maximum_average_reward=10.0,
                               evaluation_method='cumulative')
