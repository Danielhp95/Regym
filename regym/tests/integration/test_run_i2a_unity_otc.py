from regym.rl_algorithms.agents import build_I2A_Agent
from regym.rl_loops.singleagent_loops import rl_loop
from test_fixtures import i2a_config_dict, otc_task


def test_can_run_episodes_inside_otc_env(i2a_config_dict, otc_task):
    agent_i2a = build_I2A_Agent(config=i2a_config_dict, task=otc_task, agent_name='TestI2A')
    import time
    start_time = time.time()
    trajectory = rl_loop.run_episode(otc_task.env, agent_i2a, training=True, max_episode_length=100)
    print('Duration ', time.time() - start_time)
    otc_task.env.close()

    assert trajectory is not None
    assert isinstance(trajectory, list)
