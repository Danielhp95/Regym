from regym.rl_algorithms.agents import build_I2A_Agent
from regym.rl_loops.singleagent_loops import rl_loop
from test_fixtures import i2a_config_dict, otc_task
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os 
import math 
import torch

def test_train_i2a_otc_env(i2a_config_dict, otc_task):
    #logdir = './test_i2a/'
    logdir = './test_i2a_rnn/'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    sum_writer = SummaryWriter(logdir)
    save_path = os.path.join(logdir,'./i2a.agent')

    agent_i2a = build_I2A_Agent(config=i2a_config_dict, task=otc_task, agent_name='TestI2A_RNN')
    #agent_i2a = build_I2A_Agent(config=i2a_config_dict, task=otc_task, agent_name='TestI2A_MLP')
    nbr_episodes = 1e6
    max_episode_length = math.inf

    for i in tqdm(range(int(nbr_episodes))):
        trajectory = rl_loop.run_episode(otc_task.env, agent_i2a, training=True, max_episode_length=max_episode_length)
        torch.save(agent_i2a, save_path)

        total_return = sum([ t[2] for t in  trajectory])
        sum_writer.add_scalar('Training/TotalReturn', total_return, i)
        
    otc_task.env.close()

    assert trajectory is not None
    assert isinstance(trajectory, list)

if __name__ == '__main__':
    test_train_i2a_otc_env(i2a_config_dict(),otc_task())