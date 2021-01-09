from typing import List, Union, Any, Callable, Optional, Dict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from regym.rl_algorithms.agents import Agent
from regym.networks.preprocessing import parse_preprocessing_fn


class NeuralNetAgent(Agent):

    def __init__(self, neural_net: Optional[nn.Module],
                 state_preprocess_fn: Optional[Callable[[Any, List[int]], Any]],
                 name: str):
        super().__init__(name=name)
        self.neural_net = neural_net
        self.state_preprocess_fn = state_preprocess_fn

    @torch.no_grad()
    def model_free_take_action(self, observation,
                               legal_actions: List[int], multi_action: bool = False):
        observation = self.state_preprocess_fn(observation)
        self.current_prediction = self.neural_net(observation, legal_actions=legal_actions)

        action = self.current_prediction['a']
        if not multi_action:  # Action is a single integer
            action = np.int(action)
        if multi_action:  # Action comes from a vector env, one action per environment
            action = action.view(1, -1).squeeze(0).numpy()
        return action

    def handle_experience(self, o, a, r: float, succ_o, done=False):
        super().handle_experience(o, a, r, succ_o, done)

    def handle_multiple_experiences(self, experiences: List, env_ids: List[int]):
        super().handle_multiple_experiences(experiences, env_ids)

    def __repr__(self):
        return f'NeuralNetAgent: {self.name}\nModel:{self.neural_net}'

    def clone(self):
        return NeuralNetAgent(self.neural_net, self.state_preprocess_fn,
                              self.name)


def generate_preprocessing_function(state_preprocess_fn: Union[Callable, str]) \
                                    -> Callable:
    if isinstance(state_preprocess_fn, Callable): return state_preprocess_fn
    elif isinstance(state_preprocess_fn, str):
        return parse_preprocessing_fn(state_preprocess_fn)
    else: raise ValueError('Pre_processing fn should be either a Callable or a '
                           'string which can be internally parsed into a callable '
                           f"Instead {state_preprocess_fn} was given.")


def build_NeuralNet_Agent(task, config: Dict[str, Any],
                          agent_name: str) -> NeuralNetAgent:
    '''
    TODO:

    :param task: TODO
    :param config:
        - 'neural_net': (torch.nn.Module) representing the policy for this agent
                        the neural_net will be deepcopied.
        - 'state_preprocess_fn': (Callable) function used to format environment
                               observations into something usable by the
                               neural net
    :param agent_name: String identifier
    '''
    check_input_validity(config)
    neural_net = deepcopy(config['neural_net'])
    state_preprocess_fn = generate_preprocessing_function(config['preprocessing_fn'])
    return NeuralNetAgent(neural_net, state_preprocess_fn, agent_name)


def check_input_validity(config):
    '''
    TODO:
        - check neural net (1) exists (2) it is of the right type (nn.module)
        - Check state_preprocess_fn is callable
    '''
    pass
