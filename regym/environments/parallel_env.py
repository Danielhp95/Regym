import gym
import numpy as np 
import copy
import time 
from .utils import EnvironmentCreationFunction

import torch
#torch.multiprocessing.set_sharing_strategy('file_system')
from torch.multiprocessing import Process, Queue


def env_worker(envCreationFunction, queue_in, queue_out):
    continuer = True
    env = envCreationFunction()     
    
    while continuer:
        instruction = queue_in.get()

        if isinstance(instruction,bool):
            continuer = False
        elif isinstance(instruction, str):
            observations = env.reset()
            queue_out.put(observations)
        else :
            pa_a = instruction
            obs, r, done, info = env.step( pa_a)
            
            queue_out.put( [obs,r,done,info] )

    env.close()



class ParallelEnv():
    def __init__(self, envname, nbr_parallel_env):
        self.envname = envname
        self.nbr_parallel_env = nbr_parallel_env
        self.env_processes = None

    def get_nbr_envs(self):
        return self.nbr_parallel_env

    def reset(self) :
        if self.env_processes is None :
            self.env_queues = [ {'in':Queue(), 'out':Queue()} for _ in range(self.nbr_parallel_env)]
            self.env_processes = [ Process(target=env_worker, args=(EnvironmentCreationFunction(self.envname), *queues.values(),) ) for queues in self.env_queues]
            
            for idx, p in enumerate(self.env_processes):
                p.start()
                
        for idx, p in enumerate(self.env_processes):
            self.env_queues[idx]['in'].put('reset')

        observations = [ queues['out'].get() for queues in self.env_queues]
        
        per_env_obs = [ np.concatenate( [ np.array(obs[idx_agent]).reshape(1,-1) for obs in observations], axis=0) for idx_agent in range(len(observations[0]) ) ]
        self.dones = [False]*self.nbr_parallel_env
        self.previous_dones = copy.deepcopy(self.dones)
        return per_env_obs

    def step(self, action_vector):
        observations = []
        rewards = []
        infos = []
    	
        batch_env_index = -1
        for env_index in range(len(self.env_queues) ):
            if self.dones[env_index] and self.previous_dones[env_index]:
                infos.append(None)
                continue
            batch_env_index += 1
            
            pa_a = [ action_vector[idx_agent][batch_env_index] for idx_agent in range( len(action_vector) ) ]
            
            self.env_queues[env_index]['in'].put(pa_a)

        for env_index in range(len(self.env_queues) ):
            if self.dones[env_index] and self.previous_dones[env_index]:
                continue
            self.previous_dones[env_index] = self.dones[env_index] 
            
            experience = self.env_queues[env_index]['out'].get()

            obs, r, done, info = experience

            observations.append( obs )
            rewards.append( r )
            self.dones[env_index] = done
            infos.append(info)
        
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