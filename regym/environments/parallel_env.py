import gym
import numpy as np 
import copy
import time 
from .utils import EnvironmentCreator

import torch
# https://github.com/pytorch/pytorch/issues/11201:
#torch.multiprocessing.set_sharing_strategy('file_system')
from torch.multiprocessing import Process, Queue


def env_worker(env, queue_in, queue_out, worker_id=None):
    continuer = True
    #env = envCreator(worker_id=worker_id)     
    
    obs = None
    r = None
    done = None
    info = None

    try:
        while continuer:
            instruction = queue_in.get()

            if isinstance(instruction,bool):
                continuer = False
            elif isinstance(instruction, str) or isinstance(instruction, tuple):
                env_config = instruction[-1]
                if env_config is None: observations = env.reset()
                else:  observations = env.reset(env_config) 
                done = False 
                queue_out.put(observations)
            else :
                if not(done): 
                    pa_a = instruction
                    obs, r, done, info = env.step( pa_a)
                
                queue_out.put( [obs,r,done,info] )
    finally:
        env.close()


class ParallelEnv():
    def __init__(self, env_creator, nbr_parallel_env, nbr_frame_stacking=1, single_agent=True):
        self.env_creator = env_creator
        self.nbr_parallel_env = nbr_parallel_env
        self.nbr_frame_stacking = nbr_frame_stacking
        self.env_processes = None
        self.single_agent = single_agent

    def get_nbr_envs(self):
        return self.nbr_parallel_env

    def reset(self, env_configs=None) :
        if self.env_processes is None :
            self.env_queues = [ {'in':Queue(), 'out':Queue()} for _ in range(self.nbr_parallel_env)]
            worker_ids = [None]*self.nbr_parallel_env
            if env_configs is not None: 
                worker_ids = [ env_config.pop('worker_id', None) for env_config in env_configs]
            
            '''
            self.env_processes = [ Process(target=env_worker, args=(self.env_creator, *queues.values(), worker_id) ) for queues, worker_id in zip(self.env_queues, worker_ids)]
            
            for idx, p in enumerate(self.env_processes):
                p.start()
                print('Launching environment {}...'.format(idx))
                time.sleep(2)
            '''
            self.env_processes = []
            for queues, worker_id in zip(self.env_queues, worker_ids):
                p = Process(target=env_worker, args=(self.env_creator(worker_id=worker_id), *queues.values(), worker_id) )
                p.start()
                print('Launching environment ...')
                time.sleep(2)
                self.env_processes.append(p)

                
        for idx, p in enumerate(self.env_processes):
            if env_configs is None: env_config = None
            else: 
                env_config = env_configs[idx] 
                if 'worker_id' in env_config: env_config.pop('worker_id')
            self.env_queues[idx]['in'].put( ('reset', env_config))

        observations = [ queues['out'].get() for queues in self.env_queues]
        
        if self.single_agent:
            observations = [ np.concatenate([obs]*self.nbr_frame_stacking, axis=-1) for obs in observations]
            per_env_obs = np.concatenate( [ np.array(obs).reshape(1, *(obs.shape)) for obs in observations], axis=0)
        else:
            per_env_obs = [ np.concatenate( [ np.array(obs[idx_agent]).reshape(1, *(obs[idx_agent].shape)) for obs in observations], axis=0) for idx_agent in range(len(observations[0]) ) ]
        
        self.dones = [False]*self.nbr_parallel_env
        self.previous_dones = copy.deepcopy(self.dones)
        return per_env_obs

    def step(self, action_vector):
        observations = []
        rewards = []
        infos = []
    	
        batch_env_index = -1
        for env_index in range(len(self.env_queues) ):
            if self.dones[env_index]:
                continue
            batch_env_index += 1
            
            if self.single_agent:
                pa_a = action_vector[batch_env_index]
            else:
                pa_a = [ action_vector[idx_agent][batch_env_index] for idx_agent in range( len(action_vector) ) ]
            
            for i in range(self.nbr_frame_stacking):
                self.env_queues[env_index]['in'].put(pa_a)

        for env_index in range(len(self.env_queues) ):
            if self.dones[env_index]:
                infos.append(None)
                continue
            
            obses = []
            rs = []
            dones = []
            infs = []
            for i in range(self.nbr_frame_stacking):
                experience = self.env_queues[env_index]['out'].get()
                obs, r, done, info = experience
                obses.append(obs)
                rs.append(r)
                dones.append(done)
                infs.append(info)

            obs = np.concatenate(obses, axis=-1)
            r = sum(rs)
            done = any(dones)
            info = infs 

            observations.append( obs )
            rewards.append( r )
            self.dones[env_index] = done
            infos.append(info)
        
        self.previous_dones = copy.deepcopy(self.dones[env_index]) 
            
        if self.single_agent:
            per_env_obs = np.concatenate( [ np.array(obs).reshape(1, *(obs.shape)) for obs in observations], axis=0)
            per_env_reward = np.concatenate( [ np.array(r).reshape(-1) for r in rewards], axis=0)
        else:
            per_env_obs = [ np.concatenate( [ np.array(obs[idx_agent]).reshape(1,-1) for obs in observations], axis=0) for idx_agent in range(len(observations[0]) ) ]
            per_env_reward = [ np.concatenate( [ np.array(r[idx_agent]).reshape((-1)) for r in rewards], axis=0) for idx_agent in range(len(rewards[0]) ) ]

        return per_env_obs, per_env_reward, self.dones, infos

    def close(self) :
        # Tell the processes to terminate themselves:
        if self.env_processes is not None:
            for env_index in range(len(self.env_processes)):
                self.env_queues[env_index]['in'].put(False)
                self.env_processes[env_index].join()
                self.env_processes[env_index].terminate()
                
                self.env_queues[env_index]['in'].close()
                self.env_queues[env_index]['in'] = None
                self.env_queues[env_index]['out'].close()
                self.env_queues[env_index]['out'] = None

        self.env_processes = None
        self.env_queues = None
        

class ParallelEnvironmentCreationFunction():

    def __init__(self, environment_name_cli, nbr_parallel_env):
        valid_environments = ['RockPaperScissors-v0','RoboschoolSumo-v0','RoboschoolSumoWithRewardShaping-v0']
        if environment_name_cli not in valid_environments:
            raise ValueError("Unknown environment {}\t valid environments: {}".format(environment_name_cli, valid_environments))
        self.environment_name = environment_name_cli
        self.nbr_parallel_env = nbr_parallel_env

    def __call__(self):
        return ParallelEnv(self.environment_name, self.nbr_parallel_env)