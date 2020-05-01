from test_fixtures import reinforce_config_dict, CartPoleTask

from regym.rl_algorithms.agents import build_Reinforce_Agent
from regym.rl_loops.singleagent_loops.rl_loop import run_episode


def test_reinforce_can_take_actions_continuous_obvservation_discrete_action(CartPoleTask, reinforce_config_dict):
    reinforce_agent = build_Reinforce_Agent(CartPoleTask, reinforce_config_dict, 'Reinforce-Test')
    CartPoleTask.run_episode([reinforce_agent], training=False)


def test_learns_to_solve_cartpole(CartPoleTask, reinforce_config_dict):
    agent = build_Reinforce_Agent(CartPoleTask, reinforce_config_dict, 'Test-Reinforce')
    assert agent.training, 'Agent should be training in order to solve test environment'
    import tqdm
    progress_bar = tqdm.tqdm(range(4000))
    for _ in progress_bar:
        trajectory = CartPoleTask.run_episode([agent], training=True)
        progress_bar.set_description(f'{agent.name} in {CartPoleTask.env.spec.id}. Episode length: {len(trajectory)}')
