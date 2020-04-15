from typing import Dict, List
from math import sqrt

import numpy as np
import gym

import regym
from regym.rl_algorithms.agents import Agent
from regym.rl_algorithms.MCTS import sequential_mcts
from regym.rl_algorithms.MCTS import simultaneous_mcts


class MCTSAgent(Agent):

    def __init__(self, name: str, algorithm, selection_strat: str,
                 iteration_budget: int, rollout_budget: int,
                 exploration_constant: float, task_num_agents: int,
                 task_action_dim: int):
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
        self.rollout_budget = rollout_budget
        self.exploration_constant = exploration_constant
        self.task_num_agents = task_num_agents
        self.action_dim = task_action_dim

        # Different MCTS variations
        self.selection_strat = selection_strat
        self.policy_fn = self.random_selection_policy

        self.current_prediction: Dict = {}

    def random_selection_policy(self, obs, legal_actions: List[int]):
        num_legal_actions = len(legal_actions)
        action_probability = 1 / num_legal_actions
        return np.array([action_probability] * num_legal_actions)

    def model_based_take_action(self, env: gym.Env, player_index: int):
        action, visitations = self.algorithm(
                player_index=player_index,
                rootstate=env,
                budget=self.budget,
                rollout_budget=self.rollout_budget,
                num_agents=self.task_num_agents,
                selection_strat=self.selection_strat,
                policy_fn=self.policy_fn,
                exploration_factor=self.exploration_constant)

        child_visitations = [visitations[move_id] if move_id in visitations else 0.
                             for move_id in range(self.action_dim)]

        self.current_prediction['action'] = action
        self.current_prediction['child_visitations'] = child_visitations
        return action

    def handle_experience(self, s, a, r, succ_s, done=False):
        super(MCTSAgent, self).handle_experience(s, a, r, succ_s, done)

    def clone(self):
        cloned = MCTSAgent(name=self.name, algorithm=self.algorithm,
                           iteration_budget=self.budget,
                           rollout_budget=self.rollout_budget,
                           exploration_constant=self.exploration_constant,
                           task_num_agents=self.task_num_agents)
        return cloned

    def __repr__(self):
        s = f'MCTSAgent: {self.name}.\nBudget: {self.budget}\nRollout budget: {self.rollout_budget}\nExploration cnst: {self.exploration_constant}'
        return s


def build_MCTS_Agent(task: regym.environments.Task, config: Dict[str, object], agent_name: str) -> MCTSAgent:
    '''
    :param task: Task in which the agent will be able to act
    :param agent_name: String identifier for the agent
    :param config: Dictionary whose entries contain hyperparameters for the A2C agents:
        - 'budget': (Int) Number of iterations of the MCTS loop that will be carried
                    out before an action is selected.
        - 'rollout_budget': (Int) Number of steps to simulate during rollout_phase
        - 'selection_phase': (str) Which selection phase to use.
                             SUPPORTED: ['ucb1', 'puct']
        - 'exploration_constant_ucb1': UCB1 exploration constant
        - 'exploration_constant_puct': PUCT exploration constant
    :returns: Agent using an MCTS algorithm to act the :param: tasks's environment
    '''
    '''
    TODO: THIS SHOULD BE SOMEWHERE IN MCTS CODE
    - 'use_dirichlet': Whether to add dirichlet noise the the normalized child visitations.
                       Intuition behind why this is useful:
                       https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
    - 'dirichlet_alpha': Parameter of Dirichlet distribution
    - 'dirichlet_epsilon': Weight for deciding wether to prefer NN prior
                           probabilities or dirichlet noise
    '''
    check_config_validity(config, task)
    if task.env_type == regym.environments.EnvType.SINGLE_AGENT:
        raise NotImplementedError('MCTS does not currently support single agent environments')
    if task.env_type == regym.environments.EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
        algorithm = simultaneous_mcts.MCTS_UCT
    if task.env_type == regym.environments.EnvType.MULTIAGENT_SEQUENTIAL_ACTION:
        algorithm = sequential_mcts.MCTS_UCT


    budget = config['budget']
    rollout_budget = config['rollout_budget']
    selection_phase = config['selection_phase']
    exploration_constant = get_exploration_constant(config, selection_phase)

    agent = MCTSAgent(name=agent_name, algorithm=algorithm,
                      iteration_budget=budget,
                      rollout_budget=rollout_budget,
                      selection_strat=selection_phase,
                      exploration_constant=exploration_constant,
                      task_num_agents=task.num_agents,
                      task_action_dim=task.action_dim)
    return agent


def get_exploration_constant(config: Dict, selection_phase: str) -> float:
    if selection_phase == 'ucb1': return config['exploration_factor_ucb1']
    if selection_phase == 'puct': return config['exploration_factor_puct']


def check_config_validity(config: Dict, task: regym.environments.Task):
    if not isinstance(config['budget'], (int, np.integer)):
        raise ValueError('The hyperparameter \'budget\' should be an integer')
    if not isinstance(config['rollout_budget'], (int, np.integer)):
        raise ValueError('The hyperparameter \'rollout_budget\' should be an integer')
    if 'selection_phase' not in config or config['selection_phase'] not in ['ucb1', 'puct']:
        raise KeyError("A selection phase must be specified. Currently 'ucb1' and 'puct' are supported")
    if 'ucb1' == config['selection_phase'] and 'exploration_factor_ucb1' not in config:
       raise KeyError("If selection phase 'ucb1' is selected, a 'exploration_factor_ucb1' key must exist in config dict")
    if 'puct' == config['selection_phase'] and 'exploration_factor_puct' not in config:
        raise KeyError("If selection phase 'puct' is selected, a 'exploration_factor_puct' key must exist in config dict")
    if task.env_type == regym.environments.EnvType.MULTIAGENT_SIMULTANEOUS_ACTION \
            and 'puct' == config['selection_phase']:
        raise NotImplementedError(f'MCTSAgent does not currently support PUCT selection phase for tasks of type EnvType.MULTIAGENT_SIMULTANEOUS_ACTION, such as {task.name}')
