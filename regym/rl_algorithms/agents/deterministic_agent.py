from typing import Dict, List

from regym.rl_algorithms.agents import Agent


class DeterministicAgent(Agent):

    def __init__(self, action: int, name: str):
        super(DeterministicAgent, self).__init__(name=name)
        self.action = action

    def model_free_take_action(self, state, legal_actions: List[int], multi_action: bool = False):
        if not multi_action: return self.action
        else: return [self.action for _ in range(len(state))]

    def clone(self, training=None):
        pass

    def handle_experience(self, s, a, r, succ_s, done=False):
        pass

    def __repr__(self):
        return f'DeterministicAgent: {self.action}'


def build_Deterministic_Agent(task, config: Dict, agent_name: str):
    if task.action_type != 'Discrete':
        raise ValueError('Human agents can only act on Discrete action spaces, as input comes from keyboard')
    if 'action' not in config:
        raise ValueError("Config must specify an (int) 'action' for DeterministicAgent {agent_name}")
    action = int(config['action'])
    return DeterministicAgent(action=action, name=agent_name)
