from ..A2C import A2CAlgorithm


class A2CAgent():

    def __init__(self, name, samples_before_update, algorithm):
        '''
        :param name: String identifier for the agent
        :param samples_before_update: Number of actions the agent will take before updating
        :param algorithm: Reinforcement Learning algorithm used to update the agent's policy.
                          Contains the agent's policy, represented as a neural network.
        '''
        self.name = name
        self.training = True
        self.algorithm = algorithm

        self.samples_before_update = samples_before_update
        self.samples = []

    def handle_experience(self, s, a, r, succ_s, done=False):
        if not self.training: return
        self.samples.append((s, a, self.current_prediction['action_log_probability'],
                             r, self.current_prediction['state_value'], succ_s, done))
        if done or len(self.samples) >= self.samples_before_update:
            bootstrapped_reward = self.current_prediction['state_value'] if not done else 0
            self.algorithm.train(self.samples, bootstrapped_reward)
            self.samples = []

    def take_action(self, state):
        self.current_prediction = self.algorithm.model(state)
        return self.current_prediction['action'].item()

    def clone(self, training=None):
        return NotImplementedError(f'Clone function for  {self.__class__} \
                                     algorithm not yet supported')


def build_A2C_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param algorithm: Reinforcement Learning algorithm used to update the agent's policy.
                      Contains the agent's policy, represented as a neural network.
    :param config: Dictionary whose entries contain hyperparameters for the A2C agents:
        - 'discount_factor':       Discount factor (gamma in standard RL equations) used as a -variance / +bias tradeoff.
        - 'n_steps':               'Forward view' timesteps used to compute the Q_values used to approximate the advantage function
        - 'samples_before_update': Number of actions the agent will take before updating
        - 'learning_rate':         Learning rate for the Neural Network optimizer. Recommended: 1.0e-4
        - 'adam_eps':              Epsilon value used in denominator of Adam update computation. Recommended: 1.0e-5

    :returns: Agent using A2C algorithm to act and learn in environments
    '''
    algorithm = A2CAlgorithm(policy_model_input_dim=task.observation_dim, policy_model_output_dim=task.action_dim,
                             n_steps=config['n_steps'], discount_factor=config['discount_factor'],
                             adam_eps=config['adam_eps'], learning_rate=config['learning_rate'])
    return A2CAgent(name=agent_name, algorithm=algorithm, samples_before_update=config['samples_before_update'])
