from typing import Dict
import copy

import regym
from regym.rl_algorithms.agents import Agent
from regym.rl_algorithms.reinforce import ReinforceAlgorithm


class ReinforceAgent(Agent):

    def __init__(self, name: str, episodes_before_update: int, algorithm):
        '''
        :param name: String identifier for the agent
        :param episodes_before_update: Number of full environment episodes that will
                                       be sampled before computing a policy update.
        :param algorithm: Reinforcement Learning algorithm used to update the agent's policy.
                          Contains the agent's policy, represented as a neural network.
        '''
        super(ReinforceAgent, self).__init__(name=name)
        self.algorithm = algorithm

        self.episodes_before_update = episodes_before_update
        self.completed_episodes = 0
        self.trajectories = [[]]

    def handle_experience(self, s, a, r, succ_s, done=False):
        '''
        Processes a single 'experience' (defined by the parameters of this function),
        which is the main method of gathering data of an RL algorithm.
        NOTE: Unless this agent's 'training' flag is set to True, this function will not do anything.

        :param s:      Environment state
        :param a:      action taken by this agent at :param s: state
        :param r:      reward obtained by this agent after taking :param a: action at :param s: state
        :param succ_s: Environment state reached after after taking :param a: action at :param s: state
        :param done:   Boolean representing whether the environment episode has finished
        '''
        super(ReinforceAgent, self).handle_experience(s, a, r, succ_s, done)
        if not self.training: return
        trajectory_index = self.completed_episodes % self.episodes_before_update
        self.trajectories[trajectory_index].append((s, a, self.current_prediction['action_log_probability'], r, succ_s))
        if done:
            self.completed_episodes += 1
            if (self.completed_episodes % self.episodes_before_update) == 0:
                self.algorithm.train(self.trajectories)
                self.trajectories = []
            self.trajectories.append([])

    def take_action(self, state):
        '''
        :param state: Environment state
        :returns: Action to be executed by the environment conditioned on :param: state
        '''
        self.current_prediction = self.algorithm.model(state)
        return self.current_prediction['action'].item()

    def clone(self, training=None):
        '''
        :param training: Boolean specifying whether the newly cloned agent will be in training mode
        :returns: Deep cloned version of this agent
        '''
        clone = ReinforceAgent(name=self.name,
                               episodes_before_update=self.episodes_before_update,
                               algorithm=copy.deepcopy(self.algorithm))
        clone.training = training
        return clone


def build_Reinforce_Agent(task: regym.environments.Task, config: Dict[str, object], agent_name: str):
    '''
    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param algorithm: Reinforcement Learning algorithm used to update the agent's policy.
                      Contains the agent's policy, represented as a neural network.
    :param config: Dictionary whose entries contain hyperparameters for the A2C agents:
        - 'learning_rate':          Learning rate for the Neural Network optimizer. Recommended: 1.0e-4
        - 'episodes_before_update': Number of full environment episodes that will be sampled before computing a policy update. [1, infinity)
        - 'adam_eps':               Epsilon value used in denominator of Adam update computation. Recommended: 1.0e-5

    :returns: Agent using Reinforce algorithm to act and learn in environments
    '''
    algorithm = ReinforceAlgorithm(policy_model_input_dim=task.observation_dim, policy_model_output_dim=task.action_dim,
                                   learning_rate=config['learning_rate'], adam_eps=config['adam_eps'])
    return ReinforceAgent(name=agent_name, episodes_before_update=config['episodes_before_update'],
                          algorithm=algorithm)
