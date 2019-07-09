from regym.rl_algorithms.agents import build_PPO_Agent
from regym.rl_loops.singleagent_loops import rl_loop
from test_fixtures import ppo_config_dict, si_task
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import math
import torch

def test_train_ppo(ppo_config_dict, si_task):
    task = si_task
    logdir = './test_ppo_rnn_si/'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    sum_writer = SummaryWriter(logdir)
    save_path = os.path.join(logdir,'./ppo.agent')

    agent = build_PPO_Agent(config=ppo_config_dict, task=task, agent_name='TestPPO_RNN_SI')
    nbr_episodes = 1e2
    max_episode_length = math.inf

    for i in tqdm(range(int(nbr_episodes))):
        trajectory = rl_loop.run_episode(task.env, agent, training=True, max_episode_length=max_episode_length)
        torch.save(agent, save_path)

        total_return = sum([ t[2] for t in  trajectory])
        sum_writer.add_scalar('Training/TotalReturn', total_return, i)

    task.env.close()

    assert trajectory is not None
    assert isinstance(trajectory, list)
