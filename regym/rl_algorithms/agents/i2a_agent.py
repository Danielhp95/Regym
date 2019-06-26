import torch
from ..I2A import I2AAlgorithm


class I2AAgent():

    def __init__(self, name, algorithm, action_dim):
        '''
        :param name: String identifier for the agent
        :param samples_before_update: Number of actions the agent will take before updating
        :param algorithm: Reinforcement Learning algorithm used to update the agent's policy.
                          Contains the agent's policy, represented as a neural network.
        '''
        self.name = name
        self.algorithm = algorithm
        self.training = True

        self.action_dim = action_dim
        self.handled_experiences = 0

    def handle_experience(self, s, a, r, succ_s, done=False):
        '''
        Info that will be needed to be stored at this point:
          - state
          - action taken
          - log probabilities of all actions from actor (I2A)
          - log probabilities of distilled (rollout policy)
          - value estimation for the state
        Put in storage all
        '''
        if not self.training: return
        self.handled_experiences += 1

        non_terminal = torch.ones(1)*(1 - int(done))
        # Add current_prediction to storage
        self.algorithm.storage.add({'r': r, 'non_terminal': non_terminal, 's':  s})
        if (self.handled_experiences % self.algorithm.environment_update_horizon) == 0:
            self.algorithm.train_environment_model()
        if (self.handled_experiences % self.algorithm.policies_update_horizon) == 0:
            self.algorithm.train_policies()

    def take_action(self, state):
        '''
        TODO: call self.algorithm.actor_critic() to get:
                  - action taken
                  - log probabilities of all actions from actor (I2A)
                  - log probabilities of distilled (rollout policy)
                  - value estimation for the state from critic
        Put all in self.current_prediction
        '''
        import random
        return random.choice(range(self.action_dim))

    def clone(self, training=None):
        pass


def build_I2A_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param config: Dictionary whose entries contain hyperparameters for the A2C agents:
        - 'rollout_length': Number of steps to take in every imagined trajectory (length of imagined trajectory)
        - 'imagined_trajectories_per_step': Number of imagined trajectories to compute at each forward pass of the I2A (rephrase)
        - 'environment_update_horizon': (0, infinity) Number of timesteps that will elapse in between optimization calls.
        - 'policies_update_horizon': (0, infinity) Number of timesteps that will elapse in between optimization calls.
        - 'environment_model_learning_rate':
        - 'environment_model_adam_eps':
        - 'policies_learning_rate':
        - 'policies_adam_eps':
        - 'use_cuda': Whether or not to use CUDA to speed up training
    '''
    algorithm = I2AAlgorithm(rollout_length=config['rollout_length'],
                             imagined_trajectories_per_step=config['imagined_trajectories_per_step'],
                             policies_update_horizon=config['policies_update_horizon'],
                             environment_update_horizon=config['environment_update_horizon'],
                             environment_model_learning_rate=config['environment_model_learning_rate'],
                             environment_model_adam_eps=config['environment_model_adam_eps'],
                             policies_adam_learning_rate=config['policies_learning_rate'],
                             policies_adam_eps=config['policies_adam_eps'],
                             use_cuda=config['use_cuda'])
    return I2AAgent(algorithm=algorithm, name=agent_name, action_dim=task.action_dim)
