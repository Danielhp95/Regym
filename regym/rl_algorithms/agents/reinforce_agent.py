from ..reinforce import ReinforceAlgorithm
import numpy as np


class ReinforceAgent(object):

    def __init__(self, name, episodes_before_update, algorithm):
        self.algorithm = algorithm

        self.episodes_before_update = episodes_before_update
        self.completed_episodes = 0
        self.trajectories = [[]]

    def handle_experience(self, s, a, r, succ_s, done=False):
        self.trajectories[self.completed_episodes].append((s, a, r, succ_s))
        if done:
            self.completed_episodes += 1
            self.trajectories.append([])
            if self.episodes_before_update >= self.completed_episodes:
                self.algorithm.train(self.trajectories)
                self.trajectories = [[]]

    def take_action(self, state):
        prediction = self.algorithm.model(state)
        action = prediction['action'].numpy()
        if action.shape == (1, 1): # If action is a single integer
            action = np.int(action)
        return action

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
                                   learning_rate=config['learning_rate'])
    return ReinforceAgent(name=agent_name, episodes_before_update=config['episodes_before_update'],
                          algorithm=algorithm)
