from typing import Dict
import numpy as np
import gym

import regym
from regym.rl_algorithms.agents import Agent
from regym.rl_algorithms.MCTS import sequential_mcts
from regym.rl_algorithms.MCTS import simultaneous_mcts


class MCTSAgent(Agent):

    def __init__(self, name: str, algorithm, iteration_budget: int, task_num_agents: int):
        '''
        Agent for various algorithms of the Monte Carlo Tree Search family (MCTS).
        MCTS algorithms are model based (aka, statistical forward planners). which will require
        a copy of the environment every time MCTSAgent.take_action() is invoked.

        Currently, MCTSAgent supports Multiagent environments. Refer to
        regym.rl_algorithms.MCTS for details on algorithmic implementations.

        A nice survey paper of MCTS approaches:
                https://www.researchgate.net/publication/235985858_A_Survey_of_Monte_Carlo_Tree_Search_Methods
            '''
        super(MCTSAgent, self).__init__(name=name, requires_environment_model=True)
        self.algorithm = algorithm
        self.budget = iteration_budget
        self.task_num_agents = task_num_agents

    def take_action(self, env: gym.Env, player_index: int):
        player_actions = self.algorithm(env, self.budget, self.task_num_agents)
        return player_actions[player_index]

    def handle_experience(self, s, a, r, succ_s, done=False):
        super(MCTSAgent, self).handle_experience(s, a, r, succ_s, done)

    def clone(self):
        cloned = MCTSAgent(name=self.name, algorithm=self.algorithm,
                           iteration_budget=self.budget,
                           task_num_agents=self.task_num_agents)
        return cloned

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
    :returns: Agent using an MCTS algorithm to act the :param: tasks's environment
    '''
    if task.env_type == regym.environments.EnvType.SINGLE_AGENT:
        raise NotImplementedError('MCTS does not currently support single agent environments')
    if task.env_type == regym.environments.EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
        algorithm = simultaneous_mcts.MCTS_UCT
    if task.env_type == regym.environments.EnvType.MULTIAGENT_SEQUENTIAL_ACTION:
        algorithm = sequential_mcts.MCTS_UCT

    check_config_validity(config)

    agent = MCTSAgent(name=agent_name, algorithm=algorithm,
                      iteration_budget=config['budget'],
                      task_num_agents=task.num_agents)
    return agent


def check_config_validity(config: Dict[str, object]):
    if not isinstance(config['budget'], (int, np.integer)):
        raise ValueError('The hyperparameter \'budget\' should be an integer')
