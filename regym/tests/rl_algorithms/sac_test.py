from regym.rl_algorithms.agents import build_SAC_Agent
from regym.rl_algorithms import rockAgent

from test_fixtures import RPSTask, CartPoleTask, sac_config_dict
from regym.tests.test_utils.play_against_fixed_opponent import learn_against_fix_opponent


def test_sac_can_take_actions(RPSTask, sac_config_dict):
    env = RPSTask.env
    agent = build_SAC_Agent(RPSTask, sac_config_dict)
    number_of_actions = 30
    agent.training = False
    for i in range(number_of_actions):
        # asumming that first observation corresponds to observation space of this agent
        random_observation = env.observation_space.sample()[0]
        a = agent.model_free_take_action(random_observation, legal_actions=[0, 1, 2])
        observation, rewards, done, info = env.step([a, a])
        assert RPSTask.env.action_space.contains([a, a])


def test_sac_can_take_actions_continuous_obvservation_discrete_action(CartPoleTask, sac_config_dict):
    agent = build_SAC_Agent(CartPoleTask, sac_config_dict, 'SAC-CartPole-test')
    agent.training = False
    CartPoleTask.run_episode([agent], training=False)


def test_sac_can_take_actions_continuous_obvservation_discrete_action_using_model(CartPoleTask, sac_config_dict):
    agent = build_SAC_Agent(CartPoleTask, sac_config_dict, 'SAC-CartPole-test')
    agent.training = False
    CartPoleTask.run_episode([agent], training=False)


def test_sac_learns_to_beat_rock_in_RPS(RPSTask, sac_config_dict):
    from torch.utils.tensorboard import SummaryWriter
    import regym
    regym.rl_algorithms.SAC.soft_actor_critic_losses.summary_writer = SummaryWriter('sac_test_tensorboard')
    agent = build_SAC_Agent(RPSTask, sac_config_dict)
    assert agent.training
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0,  # Doesn't matter in RPS
                               task=RPSTask,
                               training_episodes=250,
                               test_episodes=50,
                               reward_tolerance=2.,
                               maximum_average_reward=10.0,
                               evaluation_method='cumulative')


def test_learns_to_solve_cartpole(CartPoleTask, sac_config_dict):
    agent = build_SAC_Agent(CartPoleTask, sac_config_dict)
    assert agent.training, 'Agent should be training in order to solve test environment'
    import tqdm
    progress_bar = tqdm.tqdm(range(20000))
    for _ in progress_bar:
        trajectory = CartPoleTask.run_episode([agent], training=True)
        progress_bar.set_description(f'{agent.name} in {CartPoleTask.env.spec.id}. Episode length: {len(trajectory)}')
