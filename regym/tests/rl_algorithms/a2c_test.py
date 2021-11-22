from test_fixtures import a2c_config_dict, CartPoleTask, RPSTask

from regym.rl_algorithms import rockAgent

from regym.rl_algorithms.agents import build_A2C_Agent


def test_a2c_can_take_actions_continuous_obvservation_discrete_action(CartPoleTask, a2c_config_dict):
    agent = build_A2C_Agent(CartPoleTask, a2c_config_dict, 'A2C-CartPoleTask-test')
    CartPoleTask.run_episode([agent], training=False)


def test_learns_to_solve_cartpole(CartPoleTask, a2c_config_dict):
    agent = build_A2C_Agent(CartPoleTask, a2c_config_dict, 'Test-A2C')
    assert agent.training, 'Agent should be training in order to solve test environment'
    import tqdm
    progress_bar = tqdm.tqdm(range(20000))
    for _ in progress_bar:
        trajectory = CartPoleTask.run_episode([agent], training=True)
        progress_bar.set_description(f'{agent.name} in {CartPoleTask.env.spec.id}. Episode length: {len(trajectory)}')


def test_a2c_learns_to_beat_rock_in_RPS(RPSTask, a2c_config_dict):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    from play_against_fixed_opponent import learn_against_fix_opponent

    agent = build_A2C_Agent(RPSTask, a2c_config_dict, 'A2C')
    assert agent.training
    learn_against_fix_opponent(agent, fixed_opponent=rockAgent,
                               agent_position=0, # Doesn't matter in RPS
                               task=RPSTask,
                               training_episodes=100,
                               benchmark_every_n_episodes=0,
                               test_episodes=50,
                               reward_tolerance=1.,
                               maximum_average_reward=10.0,
                               evaluation_method='cumulative')

