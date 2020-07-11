from typing import List

import torch.nn as nn

from regym.rl_algorithms.agents import Agent
from regym.rl_algorithms.A2C import A2CAlgorithm

from regym.networks.bodies import FCBody
from regym.networks.heads import CategoricalActorCriticNet

from regym.networks.preprocessing import turn_into_single_element_batch


class A2CAgent(Agent):

    def __init__(self, name: str,
                 samples_before_update: int,
                 algorithm: A2CAlgorithm):
        '''
        :param name: String identifier for the agent
        :param samples_before_update: Number of actions the agent will take before updating
        :param algorithm: Reinforcement Learning algorithm used to update the agent's policy.
                          Contains the agent's policy, represented as a neural network.
        '''
        super(A2CAgent, self).__init__(name=name)
        self.algorithm = algorithm
        self.state_preprocess_fn = turn_into_single_element_batch

        self.samples_before_update = samples_before_update
        self.samples = []

    def handle_experience(self, s, a, r, succ_s, done=False):
        super(A2CAgent, self).handle_experience(s, a, r, succ_s, done)
        if not self.training: return
        self.samples.append((s, a, self.current_prediction['log_pi_a'],
                             r, self.current_prediction['V'], succ_s, done))
        if done or len(self.samples) >= self.samples_before_update:
            bootstrapped_reward = self.current_prediction['V'] if not done else 0
            self.algorithm.train(self.samples, bootstrapped_reward)
            self.samples = []

    def model_free_take_action(self, state, legal_actions: List[int], multi_action: bool = False):
        processed_s = self.state_preprocess_fn(state)
        self.current_prediction = self.algorithm.model(processed_s)
        return self.current_prediction['a'].item()

    def clone(self, training=None):
        raise NotImplementedError('Cloning A2CAgent not supported')


def build_A2C_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param config: Dictionary whose entries contain hyperparameters for the A2C agents:
        - 'discount_factor':       Discount factor (gamma in standard RL equations) used as a -variance / +bias tradeoff.
        - 'n_steps':               'Forward view' timesteps used to compute the Q_values used to approximate the advantage function
        - 'samples_before_update': Number of actions the agent will take before updating
        - 'learning_rate':         Learning rate for the Neural Network optimizer. Recommended: 1.0e-4
        - 'adam_eps':              Epsilon value used in denominator of Adam update computation. Recommended: 1.0e-5

    :returns: Agent using A2C algorithm to act and learn in environments
    '''
    body = FCBody(task.observation_dim, hidden_units=(256,))  # TODO: remove magic number
    model = CategoricalActorCriticNet(body=body,
                                      state_dim=task.observation_dim,
                                      action_dim=task.action_dim)

    algorithm = A2CAlgorithm(model=model,
                             n_steps=config['n_steps'], discount_factor=config['discount_factor'],
                             adam_eps=config['adam_eps'], learning_rate=config['learning_rate'])
    return A2CAgent(name=agent_name, algorithm=algorithm, samples_before_update=config['samples_before_update'])
