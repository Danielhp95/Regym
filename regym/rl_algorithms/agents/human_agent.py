from regym.rl_algorithms.agents import Agent


class HumanAgent(Agent):

    def __init__(self, number_of_actions: int, name: str):
        super(HumanAgent, self).__init__(name=name)
        self.number_of_actions = number_of_actions

    def take_action(self, state):
        action = input(f'Take action for {self.name}: Choose from 0-{self.number_of_actions}: ')
        return int(action)

    def clone(self, training=None):
        pass

    def handle_experience(self, s, a, r, succ_s, done=False):
        pass


def build_Human_Agent(task, config, agent_name):
    if task.action_type != 'Discrete':
        raise ValueError('Human agents can only act on Discrete action spaces, as input comes from keyboard')
    return HumanAgent(number_of_actions=task.action_dim, name=agent_name)
