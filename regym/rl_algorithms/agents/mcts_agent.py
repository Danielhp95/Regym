from typing import Dict, List, Tuple, Callable, Any, Union
from math import sqrt
import multiprocessing
from multiprocessing import cpu_count
from multiprocessing.connection import Connection
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch
import numpy as np
import gym

import regym
from regym.rl_algorithms.agents import Agent

from regym.rl_algorithms.MCTS.selection_strategies import old_UCB1, UCB1, PUCT
from regym.rl_algorithms.MCTS import sequential_mcts
from regym.rl_algorithms.MCTS import simultaneous_mcts
from regym.networks.servers.neural_net_server import NeuralNetServerHandler


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
        self.policy_fn: Callable[[Any, List[int], int, int], List[float]] = self.random_selection_policy
        self.server_based_policy_fn: Callable[[Any, List[int], Connection], List[float]] = None

        # Function to compute a value to backpropagate through the MCTS tree
        # at the end of rollout_phase. 2 parameters: observation, legal_actions.
        # If None, the value given by the gamestate will be used
        self.evaluation_fn: Callable[[Any, List[int]], List[float]] = None
        self.server_based_evaluation_fn: Callable[[Any, List[int], Connection], List[float]] = None

        # Adding exploration to root nodes
        self.use_dirichlet = use_dirichlet
        self.dirichlet_alpha = dirichlet_alpha

        self.current_prediction: Union[Dict, Dict[int, Dict]] = {}

        # This will be set by ExpertIterationAgent, if it requires
        # access to true opponent policies during training
        # Future work can set this handler in the (not yet implemented)
        # MCTSAgent.access_other_agents(...) function
        self.opponent_server_handler: NeuralNetServerHandler = None

        # In case we have a server hosting opponent models, we don't want
        # to save it!
        self.keys_to_not_pickle += ['opponent_server_handler']


    def access_other_agents(self,
                            other_agents_vector: List[Agent],
                            task: 'Task',
                            num_envs: int):
        if len(other_agents_vector) != 1:
            raise NotImplementedError('Using opponent modelling inside MCTS '
                                      'is only supported with tasks with 2 '
                                      f'agents. Current task has {len(other_agents_vector) + 1}')
        other_agent = other_agents_vector[0]
        ''' Make this nice '''
        #if not hasattr(other_agent, 'algorithm.model'):
        #    raise ValueError('Expected to find nn.Module at '
        #                     f'{other_agent.__class__}.algorithm.model')
        other_agent_model = other_agent.neural_net
        self.opponent_server_handler = NeuralNetServerHandler(
            num_connections=num_envs,
            net=other_agent_model,
            preprocess_fn=other_agent.state_preprocess_fn
        )

    def random_selection_policy(self,
                                obs,
                                legal_actions: List[int],
                                self_player_index: int = None,
                                requested_player_index: int = None):
        if legal_actions == []: return []
        num_legal_actions = len(legal_actions)
        action_probability = 1 / num_legal_actions
        return [action_probability if a_i in legal_actions else 0.
                for a_i in range(self.action_dim)]

    def model_based_take_action(self, env: Union[gym.Env, Dict[int, gym.Env]],
                                observation, player_index: int,
                                multi_action: bool = False):
        self.current_prediction = {}
        if multi_action:
            return self.multi_action_model_based_take_action(env, observation,
                                                             player_index)
        else:
            return self.single_action_model_based_take_action(env, observation,
                                                              player_index)


    def multi_action_model_based_take_action(self, envs: Dict[int, gym.Env],
                                             observations: Dict[int, Any],
                                             player_index: int) -> List[int]:
        # Starting a new ProcessPoolExecutor on every
        # multi_action_model_based_take_action is ugly. But it is not
        # where the bottleneck lies (polling on neural_net_server is).
        with ProcessPoolExecutor(max_workers=min(len(envs), cpu_count())) as ex:

            policy_fns, evaluation_fns = \
                    self.multi_action_select_policy_and_evaluation_fns(
                        len(envs), player_index)

            futures = [ex.submit(async_search, env_i, self.algorithm,
                                 env, observations[env_i],
                                 self.budget, self.rollout_budget,
                                 self.selection_strat, self.exploration_constant,
                                 player_index, policy_fn,
                                 evaluation_fn, self.use_dirichlet,
                                 self.dirichlet_alpha, self.num_agents)
                       for (env_i, env), policy_fn, evaluation_fn
                       in zip(envs.items(), policy_fns, evaluation_fns)]

            (child_visitations,
             action_vector) = self.extract_child_visitations_and_action_vectors(
                futures)
            self.current_prediction['child_visitations'] = torch.FloatTensor(child_visitations)
            self.current_prediction['action'] = torch.tensor(action_vector)
        return action_vector

    def extract_child_visitations_and_action_vectors(self, futures):
        visitation_vector, action_vector = [], []
        for f in futures:
            i, (action, visitations, tree) = f.result()
            action_vector += [action]
            visitation_vector += [visitations]

        child_visitations = [[visitation[move_id] if move_id in visitation else 0.
                             for move_id in range(self.action_dim)]
                             for visitation in visitation_vector]
        return child_visitations, action_vector

    def multi_action_select_policy_and_evaluation_fns(self,
                                                      num_envs: int,
                                                      player_index: int) \
            -> Tuple[List[Callable], List[Callable]]:
        ''' TODO: explain logic behind server connections'''
        if self.server_handler:
            # NOTE: is it bad that we are sharing connections between
            # evaluation_fns and policy_fns? I guess not.
            policy_fns = self.generate_server_based_policy_fns(
                num_envs, player_index)

            evaluation_fns = [partial(
                self.server_based_evaluation_fn,
                connection=self.server_handler.client_connections[i])
                for i in range(num_envs)]
        else:
            policy_fns = [self.policy_fn for _ in range(num_envs)]
            evaluation_fns = [self.evaluation_fn for _ in range(num_envs)]
        return policy_fns, evaluation_fns

    def generate_server_based_policy_fns(self,
                                         num_envs: int,
                                         player_index: int) \
            -> List[Callable[[Any, List[int], int, int], List[float]]]:
        '''
        Creates a list of policy functions, each one partially applied with
        a different thread-safe connection to be used on a different process.
        These policy function connect to a separate process to be request
        a neural_net prediction.

        Policy functions have the following paremeterization:
            - Observation
            - Legal actions vector
            - Player index (in the environments' agent vector) That
              began the search
            - Requested player index. Index of player for whom the policy
              evalution is being requested.


        :returns: List of functions, one for each process, to be used inside
                  MCTS search. These can be used to (1) Compute PUCT action
                  priors and (2) As a rollout policy.
        '''
        if self.requires_acess_to_other_agents:
            # We are accessing the real policies by other agents to compute
            # action priors to be used in PUCT formula during selection phase
            assert self.opponent_server_handler, 'There should be a server!'
            policy_fns = [partial(
                self.server_based_policy_fn,
                connection=self.server_handler.client_connections[i],
                self_player_index=player_index,
                opponent_connection=self.opponent_server_handler.client_connections[i])
                for i in range(num_envs)]
        else:
            policy_fns = [partial(
                self.server_based_policy_fn,
                self_player_index=player_index,
                connection=self.server_handler.client_connections[i])
                for i in range(num_envs)]
        return policy_fns

    def single_action_model_based_take_action(self, env: gym.Env, observation,
                                              player_index: int) -> int:
        action, visitations, tree = self.algorithm(
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

        self.record_selected_action(self.current_prediction, action, visitations)
        return action

    def record_selected_action(self, prediction, action, visitations):
        # This is needed to ensure that all actions are represented
        # because :param: env won't expose non-legal actions
        child_visitations = [visitations[move_id] if move_id in visitations else 0.
                             for move_id in range(self.action_dim)]
        prediction['action'] = action
        prediction['child_visitations'] = child_visitations

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

    def close_server(self):
        if self.server_handler: self.server_handler.close_server()
        if self.opponent_server_handler: self.opponent_server_handler.close_server()

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

    use_dirichlet = (config['selection_phase'] == 'puct') and config.get('use_dirichlet', False)

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
                      dirichlet_alpha=config.get('dirichlet_alpha', None))
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
