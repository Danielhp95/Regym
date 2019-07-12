from regym.rl_algorithms.agents import build_PPO_Agent
from regym.rl_loops.singleagent_loops import rl_loop
from regym.environments import parse_environment
from test_fixtures import ppo_rnd_config_dict_ma
from tqdm import tqdm
from tensorboardX import SummaryWriter
import os 
import math 
import copy
import random
import torch

offset_worker_id = 100

def check_path_for_agent(filepath):
    #filepath = os.path.join(path,filename)
    agent = None
    offset_episode_count = 0
    if os.path.isfile(filepath):
        print('==> loading checkpoint {}'.format(filepath))
        agent = torch.load(filepath)
        offset_episode_count = agent.episode_count
        #setattr(agent, 'episode_count', offset_episode_count)
        print('==> loaded checkpoint {}'.format(filepath))
    return agent, offset_episode_count

def update_configs(env_param2range, nbr_actors):
    env_configs = list()
    tower_seed = random.choice(env_param2range['tower-seed'])
    #allowed_floors = random.choice(env_param2range['allowed-floors'])
    for a_i in range(nbr_actors):
        env_config = copy.deepcopy(env_param2range)
        env_config['worker_id'] = a_i+offset_worker_id
        for k in env_config:
            if k == 'tower-seed':
                env_config[k] = tower_seed
                continue
            '''
            elif k == 'allowed-floors':
                env_config[k] = allowed_floors
                continue
            '''
            if isinstance(env_config[k], list):
                v = random.choice(env_config[k])
                env_config[k] = v 
        env_configs.append(env_config)
    return env_configs

def test_train_ppo_rnd(ppo_rnd_config_dict_ma):
    here = os.path.abspath(os.path.dirname(__file__))
    task = parse_environment(os.path.join(here, 'ObstacleTower/obstacletower'),
                             nbr_parallel_env=ppo_rnd_config_dict_ma['nbr_actor'], 
                             nbr_frame_stacking=ppo_rnd_config_dict_ma['nbr_frame_stacking'])
    #logdir = './test_ppo_rnd256_normintrUP1e4_cnn60phi256_a1_b256_h1024_3e-4_OTC_frameskip4/'
    logdir = './test_LABC_gru_ppo_rnd64_normIntrUP1e4_cnn60phi256gru64_a8_b1024_h1024_3e-4_OTC_frameskip4/'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    sum_writer = SummaryWriter(logdir)
    save_path = os.path.join(logdir,'./ppo_rnd.agent')

    agent, offset_episode_count = check_path_for_agent(save_path)
    if agent is None: agent = build_PPO_Agent(config=ppo_rnd_config_dict_ma, task=task, agent_name='PPO_RND_OTC')
    agent.save_path = save_path
    nbr_episodes = 1e6
    max_episode_length = 1e3

    nbr_actors = ppo_rnd_config_dict_ma['nbr_actor']
    env_param2range = { 'tower-seed':       list(range(-1,101)),                #Sets the seed used to generate the tower. -1 corresponds to a random tower on every reset() call.
                        'starting-floor':   0,      #list(range(100)),          #Sets the starting floor for the agent on reset().
                        'total-floors':     100,    #list(range(1, 100))        #Sets the maximum number of possible floors in the tower.
                        'dense-reward':     0,      #(0, 1)                     #Whether to use the sparse (0) or dense (1) reward function.
                        'lighting-type':    [0, 1, 2],                          #Whether to use no realtime light (0), a single realtime light with minimal color variations (1), or a realtime light with large color variations (2).
                        'visual-theme':     0,      #[0, 1, 2],                 #Whether to use only the default-theme (0), the normal ordering or themes (1), or a random theme every floor (2).
                        'agent-perspective':1,      #(0, 1),                    #Whether to use first-person (0) or third-person (1) perspective for the agent.
                        'allowed-rooms':    2,      #(0, 1, 2),                 #Whether to use only normal rooms (0), normal and key rooms (1), or normal, key, and puzzle rooms (2).
                        'allowed-modules':  2,      #(0, 1, 2),                 #Whether to fill rooms with no modules (0), only easy modules (1), or the full range of modules (2).
                        'allowed-floors':   0,      #[0, 1, 2],                          #Whether to include only straightforward floor layouts (0), layouts that include branching (1), or layouts that include branching and circling (2).
                        'default-theme':    [0, 1, 2, 3, 4]                     #Whether to set the default theme to Ancient (0), Moorish (1), Industrial (2), Modern (3), or Future (4).
                        }
    
    # PARAMETERS with curriculum since they only include straightforward floors...

    env_configs = update_configs(env_param2range, nbr_actors)

    for i in tqdm(range(offset_episode_count, int(nbr_episodes))):
        trajectory = rl_loop.run_episode_parallel(task.env, agent, 
                                                  training=True, 
                                                  max_episode_length=max_episode_length,
                                                  env_configs=env_configs)
        
        total_return = [ sum([ exp[2] for exp in t]) for t in trajectory]
        mean_total_return = sum( total_return) / len(trajectory)
        
        total_int_return = [ sum([ exp[3] for exp in t]) for t in trajectory]
        mean_total_int_return = sum( total_int_return) / len(trajectory)
        
        for idx, (ext_ret, int_ret) in enumerate(zip(total_return, total_int_return)):
            sum_writer.add_scalar('Training/TotalReturn', ext_ret, i*len(trajectory)+idx)
            sum_writer.add_scalar('Training/TotalIntReturn', int_ret, i*len(trajectory)+idx)
        
        episode_lengths = [ len(t) for t in trajectory]
        mean_episode_length = sum( episode_lengths) / len(trajectory)
        std_episode_length = math.sqrt( sum( [math.pow( l-mean_episode_length ,2) for l in episode_lengths]) / len(trajectory) )
        
        sum_writer.add_scalar('Training/MeanTotalReturn', mean_total_return, i)
        sum_writer.add_scalar('Training/MeanTotalIntReturn', mean_total_int_return, i)
        
        sum_writer.add_scalar('Training/MeanEpisodeLength', mean_episode_length, i)
        sum_writer.add_scalar('Training/StdEpisodeLength', std_episode_length, i)

        # Update configs:
        env_configs = update_configs(env_param2range, nbr_actors)
        agent.episode_count += 1

    task.env.close()

    assert trajectory is not None
    assert isinstance(trajectory, list)


if __name__ == '__main__':
    test_train_ppo_rnd(ppo_rnd_config_dict_ma())