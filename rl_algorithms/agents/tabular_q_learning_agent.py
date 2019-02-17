import random
from ..TQL import TabularQLearningAlgorithm


class TabularQLearningAgent(TabularQLearningAlgorithm):
    def __init__(self, state_space_size, action_space_size, hashing_function, learning_rate=0.5, training=True):
        super(TabularQLearningAgent, self).__init__(state_space_size, action_space_size, hashing_function, learning_rate, training)

    def handle_experience(self, s, a, r, succ_s, done=False):
        if self.training:
            self.update_q_table(self.hashing_function(s), a, r, self.hashing_function(succ_s))

    def take_action(self, state):
        optimal_moves = self.find_optimal_moves(self.Q_table, self.hashing_function(state))
        return random.choice(optimal_moves)

    def clone(self, training=None, path=None):
        from ..agent_hook import AgentHook
        cloned = AgentHook(self, training=training, path=path)
        return cloned


def build_TabularQ_Agent(task, config):
    state_space_size, action_space_size = task.state_space_size, task.action_dim
    hash_state = task.hash_function
    return TabularQLearningAgent(state_space_size, action_space_size, hash_state,
                                 learning_rate=config['learning_rate'])
