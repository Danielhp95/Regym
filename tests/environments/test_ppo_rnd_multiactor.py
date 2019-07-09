from regym.rl_algorithms.agents import build_PPO_Agent
from regym.rl_loops.singleagent_loops import rl_loop
from regym.environments import parse_environment
from test_fixtures import ppo_rnd_config_dict_ma
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os 
import math 
import torch

def test_train_ppo_rnd(ppo_rnd_config_dict_ma):
    '''
    task = parse_environment("Pendulum-v0",nbr_parallel_env=ppo_rnd_config_dict_ma['nbr_actor'])
    logdir = './test_ppo_rnd_normOFFLINE1e15intr_mlpphi256_a64_b1024_h1024_3e-4_pendulum/'
    '''
    task = parse_environment("Atlantis-v0",nbr_parallel_env=ppo_rnd_config_dict_ma['nbr_actor'])
    logdir = './test_ppo_rnd_cnn80phi256_a64_b1024_h1024_3e-4_atlantis/'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    sum_writer = SummaryWriter(logdir)
    save_path = os.path.join(logdir,'./ppo_rnd.agent')

    agent = build_PPO_Agent(config=ppo_rnd_config_dict_ma, task=task, agent_name='TestPPO_RND')
    nbr_episodes = 1e4
    max_episode_length = 500

    for i in tqdm(range(int(nbr_episodes))):
        trajectory = rl_loop.run_episode_parallel(task.env, agent, training=True, max_episode_length=max_episode_length)
        torch.save(agent, save_path)

        total_return = sum( [ sum([ exp[2] for exp in t]) for t in trajectory]) / len(trajectory)
        total_int_return = sum( [ sum([ exp[3] for exp in t]) for t in trajectory]) / len(trajectory)
        sum_writer.add_scalar('Training/TotalReturn', total_return, i)
        sum_writer.add_scalar('Training/TotalIntReturn', total_int_return, i)
        
    task.env.close()

    assert trajectory is not None
    assert isinstance(trajectory, list)
