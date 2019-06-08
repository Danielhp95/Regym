from ..reinforce import ReinforceAlgorithm
import numpy as np


class ReinforceAgent(object):

    def __init__(self, name, episodes_before_update, algorithm):
        self.name = name
        self.training = True
        self.algorithm = algorithm

        self.episodes_before_update = episodes_before_update
        self.completed_episodes = 0
        self.trajectories = [[]]

    def handle_experience(self, s, a, r, succ_s, done=False):
        self.trajectories[self.completed_episodes].append((s, a, self.current_prediction['action_log_probability'], r, succ_s))
        if done:
            self.completed_episodes += 1
            if self.completed_episodes >= self.episodes_before_update:
                self.algorithm.train(self.trajectories)
                self.trajectories = []
                self.completed_episodes = 0
            self.trajectories.append([])

    def take_action(self, state):
        self.current_prediction = self.algorithm.model(state)
        return self.current_prediction['action'].item()

    def clone(self, training=None):
        clone = ReinforceAgent(name=self.name)
        clone.training = training
        return clone


def build_Reinforce_Agent(task, config, agent_name):
    '''

    :param task: Environment specific configuration
    :param episodes_before_update: Number of episode trajectories to be used to compute policy's utility gradient
    :param algorithm: TODO
    '''
    algorithm = ReinforceAlgorithm(policy_model_input_dim=task.observation_dim, policy_model_output_dim=task.action_dim,
                                   learning_rate=config['learning_rate'], adam_eps=config['adam_eps'])
    return ReinforceAgent(name=agent_name, episodes_before_update=config['episodes_before_update'],
                          algorithm=algorithm)
