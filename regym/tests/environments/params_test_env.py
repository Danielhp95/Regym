import gym


class ParamsTestEnv(gym.Env):
    """
    Test environment to test whether we can safely pass
    params to an environment through `regym.environments.generate_task`
    """
    def __init__(self, param1, param2, param3):
        self.param1, self.param2, self.param3 = param1, param2, param3
        # These two fields are required for our parser not to break
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(1)
