from typing import Dict, List, Callable, Any, Union
from math import sqrt
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed

import numpy as np
import gym

import regym
from regym.rl_algorithms.agents import Agent

from regym.rl_algorithms.MCTS.selection_strategies import old_UCB1, UCB1, PUCT
from regym.rl_algorithms.MCTS import sequential_mcts
from regym.rl_algorithms.MCTS import simultaneous_mcts


class MCTSAgent(Agent):

    def __init__(self, name: str, algorithm, selection_strat: str,
                 selection_phase_id: str,
                 iteration_budget: int, rollout_budget: int,
                 exploration_constant: float, task_num_agents: int,
                 task_action_dim: int, use_dirichlet: bool, dirichlet_alpha: float):
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
        self.budget: int = iteration_budget
        self.rollout_budget: int = rollout_budget
        self.exploration_constant: float = exploration_constant
        self.num_agents: int = task_num_agents
        self.action_dim: int = task_action_dim

        # Different MCTS variations
        self.selection_phase_id: str = selection_phase_id
        self.selection_strat: Callable = selection_strat
        # Function used to obtain a distribution over actions legal actions
        # given 2 parameters: observation, legal_actions.
        # In any given node. Used in PUCT selection_strat and ExpertIterationAgent
        self.policy_fn: Callable[[Any, List[int]], List[float]] = self.random_selection_policy

        # Function to compute a value to backpropagate through the MCTS tree
        # at the end of rollout_phase. 2 parameters: observation, legal_actions.
        # If None, the value given by the gamestate will be used
        self.evaluation_fn: Callable[[Any, List[int]], List[float]] = None

        # Adding exploration to root nodes
        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha

        self.current_prediction: Union[Dict, Dict[int, Dict]] = {}

    def random_selection_policy(self, obs, legal_actions: List[int]):
        if legal_actions == []: return []
        num_legal_actions = len(legal_actions)
        action_probability = 1 / num_legal_actions
        return [action_probability if a_i in legal_actions else 0.
                for a_i in range(self.action_dim)]

    def model_based_take_action(self, env: Union[gym.Env, Dict[int, gym.Env]],
                                observation, player_index: int, multi_action: bool = False):
        if multi_action:
            return self.multi_action_model_based_take_action(env, observation,
                                                             player_index)
        else:
            action, visitations = self.algorithm(
                    player_index=player_index,
                    rootstate=env,
                    observation=observation,
                    budget=self.budget,
                    rollout_budget=self.rollout_budget,
                    evaluation_fn=self.evaluation_fn,
                    num_agents=self.num_agents,
                    selection_strat=self.selection_strat,
                    policy_fn=self.policy_fn,
                    exploration_factor=self.exploration_constant,
                    use_dirichlet=self.use_dirichlet,
                    dirichlet_alpha=self.dirichlet_alpha)

            # This is needed to ensure that all actions are represented
            # because :param: env won't expose non-legal actions
            child_visitations = [visitations[move_id] if move_id in visitations else 0.
                                 for move_id in range(self.action_dim)]

            self.current_prediction['action'] = action
            self.current_prediction['child_visitations'] = child_visitations
            return action

    def multi_action_model_based_take_action(self, envs: Dict[int, gym.Env],
                                             observations: Dict[int, Any],
                                             player_index: int) -> List[int]:
        action_vector = []
        # NOTE: ordering of parameters depends on the underlying
        # function in self.algorithm
        with ProcessPoolExecutor(max_workers=min(len(envs), cpu_count())) as ex:
            futures = [ex.submit(async_search, env_i, self.algorithm,
                                 env, observations[env_i],
                                 self.budget, self.rollout_budget,
                                 self.selection_strat, self.exploration_constant,
                                 player_index, self.policy_fn,
                                 self.evaluation_fn, self.use_dirichlet,
                                 self.dirichlet_alpha, self.num_agents)
                       for env_i, env in envs.items()]
            for f in futures:
                i, (action, visitations) = f.result()
                action_vector += [action]
                child_visitations = [visitations[move_id] if move_id in visitations else 0.
                                     for move_id in range(self.action_dim)]

                self.current_prediction[i] = {}
                self.current_prediction[i]['action'] = action
                self.current_prediction[i]['child_visitations'] = child_visitations
        return action_vector

    def handle_experience(self, s, a, r, succ_s, done=False):
        super(MCTSAgent, self).handle_experience(s, a, r, succ_s, done)
        self.current_prediction.clear()

    def clone(self):
        cloned = MCTSAgent(name=self.name, algorithm=self.algorithm,
                           selection_strat=self.selection_strat,
                           selection_phase_id=self.selection_phase_id,
                           iteration_budget=self.budget,
                           rollout_budget=self.rollout_budget,
                           task_action_dim=self.action_dim,
                           task_num_agents=self.num_agents,
                           exploration_constant=self.exploration_constant,
                           use_dirichlet=self.use_dirichlet,
                           dirichlet_alpha=self.dirichlet_alpha)
        return cloned

    def __repr__(self):
        s = f'MCTSAgent: {self.name}.\nBudget: {self.budget}\nRollout budget: {self.rollout_budget}\nSelection phase: {self.selection_phase_id}\nExploration cnst: {self.exploration_constant}\nUse Direchlet noise: {self.use_dirichlet}\nDirchlet alpha: {self.dirichlet_alpha}'
        return s


def async_search(i, algorithm, *algorithm_args):
    results = algorithm(*algorithm_args)
    return i, results


def build_MCTS_Agent(task: regym.environments.Task, config: Dict[str, object], agent_name: str) -> MCTSAgent:
    '''
    :param task: Task in which the agent will be able to act
    :param agent_name: String identifier for the agent
    :param config: Dictionary whose entries contain hyperparameters for the A2C agents:
        - 'budget': (Int) Number of iterations of the MCTS loop that will be carried
                    out before an action is selected.
        - 'rollout_budget': (Int) Number of steps to simulate during rollout phase
        - 'selection_phase': (str) Which selection phase to use.
                             SUPPORTED: ['ucb1', 'puct']
        - 'exploration_factor_ucb1': UCB1 exploration constant.
                                     Only used if 'selection_phase'='ucb1'
        - 'exploration_factor_puct': PUCT exploration constant
                                     Only used if 'selection_phase'='puct'
        - 'use_dirichlet': Whether to add dirichlet noise to the priors of the
                           root node of the MCTS tree to encourage exploration.
                           Intuition behind why this is useful:
                           https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5
        - 'dirichlet_alpha': Parameter of Dirichlet distribution.
                             Only used if 'use_dirichlet' flag is set
    :returns: Agent using an MCTS algorithm to act the :param: tasks's environment
    '''
    check_config_validity(config, task)
    check_task_compatibility(task)

    if config['selection_phase'] == 'ucb1':
        selection_strat, exploration_constant = UCB1, config['exploration_factor_ucb1']
    if config['selection_phase'] == 'puct':
        selection_strat, exploration_constant = PUCT, config['exploration_factor_puct']

    if task.env_type == regym.environments.EnvType.MULTIAGENT_SIMULTANEOUS_ACTION:
        algorithm, selection_strat = simultaneous_mcts.MCTS_UCT, old_UCB1
    if task.env_type == regym.environments.EnvType.MULTIAGENT_SEQUENTIAL_ACTION:
        algorithm = sequential_mcts.MCTS

    use_dirichlet = (config['selection_phase'] == 'puct') and config['use_dirichlet']

    agent = MCTSAgent(name=agent_name,
                      algorithm=algorithm,
                      iteration_budget=config['budget'],
                      rollout_budget=config['rollout_budget'],
                      selection_strat=selection_strat,
                      selection_phase_id=config['selection_phase'],
                      exploration_constant=exploration_constant,
                      task_num_agents=task.num_agents,
                      task_action_dim=task.action_dim,
                      use_dirichlet=use_dirichlet,
                      dirichlet_alpha=config['dirichlet_alpha'])
    return agent


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


def check_task_compatibility(task: regym.environments.Task):
    if task.env_type == regym.environments.EnvType.SINGLE_AGENT:
        raise NotImplementedError('MCTS does not currently support single agent environments')
