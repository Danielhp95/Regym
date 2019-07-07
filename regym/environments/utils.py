import gym

class EnvironmentCreator():
    def __init__(self, environment_name_cli, is_unity_environment, is_gym_environment):
        self.environment_name = environment_name_cli
        self.is_unity_environment = is_unity_environment
        self.is_gym_environment = is_gym_environment

    def __call__(self, worker_id=None):
        if self.is_gym_environment: return gym.make(self.environment_name)
        if self.is_unity_environment: 
            if 'obstacletower' not in env_name: raise ValueError('Only obstacletower environment currently supported')
            from obstacle_tower_env import ObstacleTowerEnv
            if worker_id is None: worker_id=0
            return ObstacleTowerEnv(env_name, retro=True, realtime_mode=False, worker_id=worker_id) # retro=True mode creates an observation space of a 64x64 (Box) image
    