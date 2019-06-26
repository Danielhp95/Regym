from test_fixtures import i2a_config_dict, CartPoleTask
from utils import can_act_in_environment
from regym.rl_algorithms.agents import build_I2A_Agent
from regym.rl_loops.singleagent_loops.rl_loop import run_episode


def test_i2a_can_take_actions_continuous_obvservation_discrete_action(CartPoleTask, i2a_config_dict):
    can_act_in_environment(CartPoleTask, build_I2A_Agent, i2a_config_dict, name=__name__)


def test_i2a_learns_to_solve_cartpole(CartPoleTask, i2a_config_dict):
    agent = build_I2A_Agent(CartPoleTask, i2a_config_dict, 'Test-I2A')
    assert agent.training, 'Agent should be training in order to solve test environment'
    import tqdm
    progress_bar = tqdm.tqdm(range(200))
    for _ in progress_bar:
        trajectory = run_episode(CartPoleTask.env, agent, training=True)
        progress_bar.set_description(f'{agent.name} in {CartPoleTask.env.spec.id}. Episode length: {len(trajectory)}')
