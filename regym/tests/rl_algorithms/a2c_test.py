from test_fixtures import a2c_config_dict, CartPoleTask

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
