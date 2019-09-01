from test_fixtures import a2c_config_dict, CartPoleTask
from utils import can_act_in_environment

from regym.rl_algorithms.agents import build_A2C_Agent
from regym.rl_loops.singleagent_loops.rl_loop import run_episode


def test_a2c_can_take_actions_continuous_obvservation_discrete_action(CartPoleTask, a2c_config_dict):
    can_act_in_environment(CartPoleTask, build_A2C_Agent, a2c_config_dict, name=__name__)


def test_learns_to_solve_cartpole(CartPoleTask, a2c_config_dict):
    agent = build_A2C_Agent(CartPoleTask, a2c_config_dict, 'Test-A2C')
    assert agent.training, 'Agent should be training in order to solve test environment'
    import tqdm
    progress_bar = tqdm.tqdm(range(20000))
    for _ in progress_bar:
        trajectory = run_episode(CartPoleTask.env, agent, training=True)
        progress_bar.set_description(f'{agent.name} in {CartPoleTask.env.spec.id}. Episode length: {len(trajectory)}')
