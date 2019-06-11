from ..A2C import A2CAlgorithm


class A2CAgent():

    def __init__(self, name, samples_before_update, algorithm):
        self.name = name
        self.training = True
        self.algorithm = algorithm

        self.samples_before_update = samples_before_update
        self.samples = []

    # TODO: Check that we are correctly storing information and resetting it
    def handle_experience(self, s, a, r, succ_s, done=False):
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
        pass


def build_A2C_Agent(task, config, agent_name):
    '''
    TODO
    '''
    algorithm = A2CAlgorithm(policy_model_input_dim=task.observation_dim, policy_model_output_dim=task.action_dim,
                             k_steps=config['k_steps'], discount_factor=config['discount_factor'],
                             adam_eps=config['adam_eps'], learning_rate=config['learning_rate'])
    return A2CAgent(name=agent_name, algorithm=algorithm, samples_before_update=config['samples_before_update'])
