from typing import List, Dict

import gym
import torch.nn as nn

from regym.rl_algorithms.agents import Agent, build_MCTS_Agent, MCTSAgent
from regym.rl_algorithms.agents import Agent, MCTSAgent


class ExpertIterationAgent(Agent):

    def __init__(self, name: str,
                 expert: MCTSAgent,
                 apprentice: nn.Module = None):
        '''
        :param name: String identifier for the agent
        :param expert: Agent used to take actions in the environment
                       and create optimization targets for the apprentice
        :param apprentice: 
        '''
        super(ExpertIterationAgent, self).__init__(name=name,
                                                   requires_environment_model=True)
        self.expert = expert
        self.apprentice = apprentice


    def handle_experience(self, s, a, r, succ_s, done=False):
        super(ExpertIterationAgent, self).handle_experience(s, a, r, succ_s, done)

    def take_action(self, env: gym.Env, player_index: int):
        # Use expert to get an action / node visitation
        action = self.expert.take_action(env, player_index)
        # Store action in storage (check Daniel's implementation)
        # return action
        return action

    def clone(self):
        raise NotImplementedError('Cloning ExpertIterationAgent not supported')


def choose_feature_extractor(task, config: Dict):
    return None


def build_apprentice_model(task, config: Dict) -> nn.Module:
    feature_extractor_module = choose_feature_extractor(task, config):

def build_expert(task, config: Dict, expert_name: str) -> MCTSAgent:
    expert_config = {'budget': config['mcts_budget'],
                     'rollout_budget': config['mcts_rollout_budget']}
    return build_MCTS_Agent(task, config, agent_name=expert_name)


def build_ExpertIteration_Agent(task, config, agent_name):
    '''
    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param config: Dict contain hyperparameters for the ExpertIterationAgent:

        - 'mcts_budget': (Int) Number of iterations of the MCTS loop that will be carried
                                 out before an action is selected.
        - 'mcts_rollout_budget': (Int) Number of steps to simulate during
                                 rollout_phase
        - 'use_agent_modelling: (Bool) whether to model agent's policies as in DPIQN paper
    '''

    expert = build_expert(task, config, expert_name=f'Expert:{agent_name}')
    return ExpertIterationAgent(name=agent_name,
                                expert=expert)
