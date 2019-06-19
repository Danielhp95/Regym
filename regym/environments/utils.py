import gym

class EnvironmentCreationFunction():

    def __init__(self, environment_name_cli):
        valid_environments = ['RockPaperScissors-v0','RoboschoolSumo-v0','RoboschoolSumoWithRewardShaping-v0']
        if environment_name_cli not in valid_environments:
            raise ValueError("Unknown environment {}\t valid environments: {}".format(environment_name_cli, valid_environments))
        self.environment_name = environment_name_cli

    def __call__(self):
        return gym.make(self.environment_name)
