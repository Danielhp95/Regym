from typing import Dict
import numpy as np

from regym.rl_algorithms.MCTS import MCTS_UCT
from regym.environments.task import Task, EnvType
from regym.rl_algorithms.MCTS import MCTS_UCT


class MCTSAgent():

    def __init__(self, name, iteration_budget: int):
        '''
        MCTS is a model based algorithm, which will require
        a copy of the environment every time MCTSAgent.take_action()
        is invoked
        '''
        self.requires_environment_model = True
        self.name = name
        self.budget = iteration_budget

    def take_action(self, env):
        return MCTS_UCT(env, self.budget)

    def handle_experience(self, s, a, r, succ_s, done=False):
        pass

    def clone(self):
        pass

    def __repr__(self):
        s = f'MCTSAgent: {self.name}. Budget: {self.budget}'
        return s

def build_MCTS_Agent(task: Task, config: Dict[str, object], agent_name: str):
    if task.env_type == EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
        raise NotImplementedError('MCTS does not currently support Simultaenous multiagent environments')
    if task.env_type == EnvType.SINGLE_AGENT:
        raise NotImplementedError('MCTS does not currently support single agent environments')

    check_config_validity(config)

    # TODO: box environments are considered continuous.
    # Update so that if (space.dtype == an int type), then the space is considered discrete
    agent = MCTSAgent(name=agent_name, iteration_budget=config['budget']) 
    return agent


def check_config_validity(config: Dict[str, object]):
    if not isinstance(config['budget'], (int, np.integer)):
        raise ValueError('The hyperparameter \'budget\' should be an integer')
