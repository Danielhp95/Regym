from typing import Dict
import numpy as np
import gym

import regym
from regym.rl_algorithms.agents import Agent
from regym.rl_algorithms.MCTS import sequential_mcts
from regym.rl_algorithms.MCTS import simultaneous_mcts


class MCTSAgent(Agent):

    def __init__(self, name: str, algorithm, iteration_budget: int):
        '''
        MCTS is a model based algorithm, which will require
        a copy of the environment every time MCTSAgent.take_action()
        is invoked
        '''
        super(MCTSAgent, self).__init__(name=name, requires_environment_model=True)
        self.algorithm = algorithm
        self.budget = iteration_budget

    def take_action(self, env: gym.Env, player_index: int):
        all_player_actions = self.algorithm(env, self.budget)
        return all_player_actions[player_index]

    def handle_experience(self, s, a, r, succ_s, done=False):
        super(MCTSAgent, self).handle_experience(s, a, r, succ_s, done)

    def clone(self):
        pass

    def __repr__(self):
        s = f'MCTSAgent: {self.name}. Budget: {self.budget}'
        return s


def build_MCTS_Agent(task: regym.environments.Task, config: Dict[str, object], agent_name: str) -> MCTSAgent:
    '''
    :param task: Task in which the agent will be able to act
    :param agent_name: String identifier for the agent
    :param config: Dictionary whose entries contain hyperparameters for the A2C agents:
        - 'budget': (Int) Number of iterations of the MCTS loop that will be carried
                    out before an action is selected.
    :returns: Agent using Reinforce algorithm to act and learn in environments
    '''
    if task.env_type == regym.environments.EnvType.SINGLE_AGENT:
        raise NotImplementedError('MCTS does not currently support single agent environments')
    if task.env_type == regym.environments.EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
        algorithm = simultaneous_mcts.MCTS_UCT
    if task.env_type == regym.environments.EnvType.MULTIAGENT_SEQUENTIAL_ACTION:
        algorithm = sequential_mcts.MCTS_UCT

    check_config_validity(config)

    # TODO: box environments are considered continuous.
    # Update so that if (space.dtype == an int type), then the space is considered discrete
    agent = MCTSAgent(name=agent_name, algorithm=algorithm, iteration_budget=config['budget']) 
    return agent


def check_config_validity(config: Dict[str, object]):
    if not isinstance(config['budget'], (int, np.integer)):
        raise ValueError('The hyperparameter \'budget\' should be an integer')
