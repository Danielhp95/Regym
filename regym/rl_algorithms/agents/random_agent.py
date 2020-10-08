from typing import List, Optional, Union, Any
import random

import numpy as np
import gym

from regym.environments import EnvType
from regym.rl_algorithms.agents import Agent


class RandomAgent(Agent):
    '''
    Mostly useful for debugging purposes.

    Agent that behaves randomly in an environment by sampling random actions
    from the environment's underlying action space. It will never sample an invalid
    action if a list of legal actions is provided.
    '''

    def __init__(self, name: str, action_space: gym.spaces.Space, action_space_dim: int):
        super(RandomAgent, self).__init__(name, requires_environment_model=False)
        self.action_space = action_space
        self.action_space_dim = action_space_dim

    def model_free_take_action(self, observations: Union[Any, List[Any]],
                               legal_actions: Optional[Union[List[int], List[List[int]]]] = None,
                               multi_action: bool = False):
        self.current_prediction = {}
        if multi_action:
            action = self.compute_multi_action_and_propagate_current_prediction(
                observations=observations,
                num_actions=len(legal_actions),
                legal_actions=legal_actions
            )
        if not multi_action:
            if legal_actions: action = random.choice(legal_actions)
            else: action = self.action_space.sample()
            self.current_prediction['a'] = action
            self.current_prediction['probs'] = [1 / self.action_space_dim for _ in range(self.action_space_dim)]
            return action
        return action

    def compute_multi_action_and_propagate_current_prediction(self,
                          observations, num_actions, legal_actions) -> List[int]:
        if legal_actions:
            actions = [random.choice(legal_actions[i]) if legal_actions[i] else
                       self.action_space.sample()
                       for i in range(len(legal_actions))]
        else:
            actions = [self.action_space.sample()
                       for _ in range(len(observations))]
        self.current_prediction['a'] = actions
        self.current_prediction['probs'] = [[1 / self.action_space_dim for _ in range(self.action_space_dim)]
                                            for _ in range(len(observations))]
        return actions


    def handle_experience(self, s, a, r, succ_s, done=False):
        self.current_prediction.clear()
        super(RandomAgent, self).handle_experience(s, a, r, succ_s, done)

    def handle_multiple_experiences(self, experiences: List, env_ids: List[int]):
        self.current_prediction.clear()
        super().handle_multiple_experiences(experiences, env_ids)

    def clone(self):
        return RandomAgent(name=self.name, action_space=self.action_space)

    def __repr__(self):
        return f'{self.name}. Action space: {self.action_space}'


def build_Random_Agent(task, config, agent_name: str):
    '''
    Builds an agent that is able to randomly act in a task

    :param task: Task in which the agent will be able to act
    :param config: Ignored, left here to keep `build_X_Agent` interface consistent
    :param name: String identifier
    '''
    if task.env_type == EnvType.SINGLE_AGENT: action_space = task.env.action_space
    # Assumes all agents share same action space
    else: action_space = task.env.action_space.spaces[0]
    return RandomAgent(name=agent_name, action_space=action_space, action_space_dim=task.action_dim)
