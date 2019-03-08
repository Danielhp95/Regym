import gym
import numpy as np 
import copy

class ParallelEnv():
    def __init__(self, envname, nbr_parallel_env):
        self.envname = envname
        self.nbr_parallel_env = nbr_parallel_env
        self.envs = []

    def get_nbr_envs(self):
        return self.nbr_parallel_env

    def reset(self) :
        self.envs = [ gym.make(self.envname) for _ in range(self.nbr_parallel_env)]
        observations = [ env.reset() for env in self.envs]
        per_env_obs = [ np.concatenate( [ np.array(obs[idx_agent]).reshape(1,-1) for obs in observations], axis=0) for idx_agent in range(len(observations[0]) ) ]
        self.dones = [False]*self.nbr_parallel_env
        self.previous_dones = copy.deepcopy(self.dones)
        return per_env_obs

    def step(self, action_vector):
        observations = []
        rewards = []
        infos = []
    	
        batch_env_index = -1
        for env_index in range(len(self.envs) ):
            if self.dones[env_index] and self.previous_dones[env_index]:
                infos.append(None)
                continue
            self.previous_dones[env_index] = self.dones[env_index] 

            batch_env_index += 1
            
            pa_a = [ action_vector[idx_agent][batch_env_index] for idx_agent in range( len(action_vector) ) ]
            
            obs, r, done, info = self.envs[env_index].step( pa_a)
            
            observations.append( obs )
            rewards.append( r )
            self.dones[env_index] = done
            infos.append(info)
        
        per_env_obs = [ np.concatenate( [ np.array(obs[idx_agent]).reshape(1,-1) for obs in observations], axis=0) for idx_agent in range(len(observations[0]) ) ]
        per_env_reward = [ np.concatenate( [ np.array(r[idx_agent]).reshape((-1)) for r in rewards], axis=0) for idx_agent in range(len(rewards[0]) ) ]

        return per_env_obs, per_env_reward, self.dones, infos