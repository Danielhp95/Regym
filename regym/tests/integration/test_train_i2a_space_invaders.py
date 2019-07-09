from regym.rl_algorithms.agents import build_I2A_Agent
from regym.rl_loops.singleagent_loops import rl_loop
from test_fixtures import i2a_config_dict, si_task
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os 
import math 
import torch

def test_train_i2a(i2a_config_dict, si_task):
    task = si_task
    logdir = './test_i2a_rnn_si/'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    sum_writer = SummaryWriter(logdir)
    save_path = os.path.join(logdir,'./i2a.agent')

    agent_i2a = build_I2A_Agent(config=i2a_config_dict, task=task, agent_name='TestI2A_RNN_SI')
    import ipdb; ipdb.set_trace()
    nbr_episodes = 1e6
    max_episode_length = math.inf

    for i in tqdm(range(int(nbr_episodes))):
        trajectory = rl_loop.run_episode(task.env, agent_i2a, training=True, max_episode_length=max_episode_length)
        torch.save(agent_i2a, save_path)

        total_return = sum([ t[2] for t in  trajectory])
        sum_writer.add_scalar('Training/TotalReturn', total_return, i)
        
    task.env.close()

    assert trajectory is not None
    assert isinstance(trajectory, list)
