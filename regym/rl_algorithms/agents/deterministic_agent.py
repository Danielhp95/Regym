from typing import Dict, List, Union

import numpy as np

from regym.rl_algorithms.agents import Agent


class DeterministicAgent(Agent):
    '''
    TODO
    '''

    def __init__(self,
                 action: int,
                 action_sequence: List[int],
                 action_space_dim: int,
                 name: str = 'DeterministicAgent'):
        super(DeterministicAgent, self).__init__(name=name)

        self.repeat_single_action = bool(action is not None)

        self.action = action

        self.action_sequence = action_sequence
        self.action_sequence_i = 0

        self.action_space_dim = action_space_dim

    def model_free_take_action(self, state, legal_actions: List[int], multi_action: bool = False):
        self.current_prediction = {}
        if self.repeat_single_action:
            action = self.compute_action_from_fixed_single_action_and_populate_current_prediction(
                num_actions=len(state),
                multi_action=multi_action
            )
        else:
            action = self.compute_action_from_action_sequence_and_populate_current_prediction(
                num_actions=len(state),
                multi_action=multi_action
            )
        return action

    def compute_action_from_fixed_single_action_and_populate_current_prediction(self,
                      num_actions: int, multi_action: bool) -> Union[int, List[int]]:
        '''
        TODO: also, refactor the computation of current_prediction?
        '''
        if multi_action:
            action = [self.action for _ in range(num_actions)]

            probs = np.zeros((num_actions, self.action_space_dim))
            for i in range(num_actions): probs[i][action] = 1.
            self.current_prediction['probs'] = probs

        else:
            action = self.action
            probs = np.zeros((1, self.action_space_dim))
            probs[0][action] = 1.
            self.current_prediction['probs'] = probs
        self.current_prediction['a'] = action
        return action

    def compute_action_from_action_sequence_and_populate_current_prediction(self,
                      num_actions: int, multi_action: bool) -> Union[int, List[int]]:
        if multi_action:
            action = [self.action_sequence[self.action_sequence_i] for _ in range(len(state))]
        else:
            action = self.action_sequence[self.action_sequence_i]
        self.action_sequence_i = (self.action_sequence_i + 1) % len(self.action_sequence)
        self.current_prediction['a'] = action
        return action

    def clone(self, training=None):
        cloned_agent = DeterministicAgent(
            action=self.action,
            action_sequence=self.action_sequence,
            name=self.name)

        return cloned_agent

    def __repr__(self):
        if self.repeat_single_action:
            action_str = f'(action) {self.action}'
        else:
            action_str = f'(action sequence, i: {self.action_sequence_i}) {self.action_sequence}'
        return f'DeterministicAgent: {action_str}'


def build_Deterministic_Agent(task, config: Dict, agent_name: str):
    '''
    TODO

    Note: The same action_sequence_i is used to choose an action in all
    environments. They are not reset at the the end of an episode
    '''
    if task.action_type != 'Discrete':
        raise ValueError('Human agents can only act on Discrete action spaces, as input comes from keyboard')
    if 'action' not in config and 'action_sequence' not in config:
        raise ValueError("Config must specify an either: \n-(int) 'action' or\n-List[int] action_sequence for DeterministicAgent {agent_name}")
    if 'action' in config and 'action_sequence' in config:
        raise ValueError("Only one of the two keys must be present: 'action', 'action_sequence'")
    # TODO: clean up
    action, action_sequence = None, None
    if 'action' in config: action = int(config.get('action'))
    if 'action_sequence' in config: action_sequence = config.get('action_sequence')
    return DeterministicAgent(action=action,
                              action_sequence=action_sequence,
                              action_space_dim=task.action_dim,
                              name=agent_name)
