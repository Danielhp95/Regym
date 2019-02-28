import os
import sys
sys.path.append(os.path.abspath('..'))

from torch.multiprocessing import Process, Queue
from rl_algorithms import build_PPO_Agent, build_DQN_Agent
from environments import parse_gym_environment
from multiagent_loops.simultaneous_action_rl_loop import run_episode


def RPSenv():
    import gym
    import gym_rock_paper_scissors
    return gym.make('RockPaperScissors-v0')


def RPSTask(env):
    return parse_gym_environment(env)


def ppo_config_dict():
    config = dict()
    config['discount'] = 0.99
    config['use_gae'] = False
    config['use_cuda'] = False
    config['gae_tau'] = 0.95
    config['entropy_weight'] = 0.01
    config['gradient_clip'] = 5
    config['optimization_epochs'] = 10
    config['mini_batch_size'] = 256
    config['ppo_ratio_clip'] = 0.2
    config['learning_rate'] = 3.0e-4
    config['adam_eps'] = 1.0e-5
    config['horizon'] = 1024
    return config


def dqn_config_dict():
    config = dict()
    config['batch_size'] = 32
    config['gamma'] = 0.99
    config['tau'] = 1.0e-3
    config['learning_rate'] = 1.0e-3
    config['epsstart'] = 0.8
    config['epsend'] = 0.05
    config['epsdecay'] = 1.0e3
    config['double'] = False
    config['dueling'] = False
    config['use_cuda'] = False
    config['use_PER'] = False
    config['PER_alpha'] = 0.07
    config['min_memory'] = 5.0e1
    config['memoryCapacity'] = 25.0e3
    config['nbrTrainIteration'] = 32
    return config


def self_play_training_process(queue, end_queue):
    env = RPSenv()

    agent = queue.get()
    agent2 = queue.get()

    run_episode(env, [agent, agent2], training=False)
    
    end_queue.put(agent)
    end_queue.put(agent2)

    while True:
        pass 


if __name__ == '__main__':
    import torch
    torch.multiprocessing.set_start_method('forkserver', force=True)

    agent  = build_PPO_Agent(RPSTask(RPSenv()), ppo_config_dict())
    agent2  = build_PPO_Agent(RPSTask(RPSenv()), ppo_config_dict())
    # agent2 = build_DQN_Agent(RPSTask(RPSenv()), dqn_config_dict())
    agent_queue = Queue()
    end_queue = Queue()

    p = Process(target=self_play_training_process, args=(agent_queue, end_queue))

    agent_queue.put(agent)
    agent_queue.put(agent2)
    p.start()
    
    print(end_queue.qsize())
    end_agent = end_queue.get()
    print(end_agent)
    
    end_agent2 = end_queue.get()
    print(end_agent2)