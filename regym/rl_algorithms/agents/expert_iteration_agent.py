from typing import List, Dict, Any, Callable, Union, Tuple, Optional
from multiprocessing.connection import Connection
from functools import partial
import multiprocessing
import textwrap

import gym
import numpy as np
import torch
import torch.nn as nn

from regym.rl_algorithms.replay_buffers import Storage

from regym.networks import Convolutional2DBody, FCBody, CategoricalActorCriticNet, SequentialBody, PolicyInferenceActorCriticNet
from regym.networks.preprocessing import (turn_into_single_element_batch,
                                          batch_vector_observation,
                                          parse_preprocessing_fn)

from regym.rl_algorithms.agents import Agent, build_MCTS_Agent, MCTSAgent

from regym.rl_algorithms.expert_iteration import ExpertIterationAlgorithm

from regym.networks.servers.neural_net_server import NeuralNetServerHandler
from regym.networks.servers import request_prediction_from_server


class ExpertIterationAgent(Agent):

    def __init__(self, algorithm: ExpertIterationAlgorithm,
                 name: str,
                 expert: MCTSAgent, apprentice: nn.Module,
                 use_apprentice_in_expert: bool,
                 use_agent_modelling: bool,
                 use_true_agent_models_in_mcts: bool,
                 use_learnt_opponent_models_in_mcts: bool,
                 action_dim: int,
                 observation_dim: Tuple[int],
                 num_opponents: int,
                 state_preprocess_fn: Optional[Callable],
                 server_state_preprocess_fn: Optional[Callable],
                 use_cuda: bool):
        '''
        :param algorithm: ExpertIterationAlgorithm which will be fed
                          trajectories computed by this agent, in turn
                          this will update the parameters of :param: apprentice
        :param name: String identifier for the agent
        :param expert: Agent used to take actions in the environment
                       and create optimization targets for the apprentice
        :param apprentice: Neural network used inside of the expert.
                           It will be used to compute priors for MCTS nodes
                           and values to backpropagate.
                           This neural net will be changed by :param: algorithm
        :param use_apprentice_in_expert: whether to bias MCTS's selection
                                         phase and expansion phase with the
                                         apprentice. If False, this algorithm
                                         is equivalent to DAGGER
        :param use_agent_modelling: Wether to model other agent's actions as
                                    an auxiliary task. As in DPIQN paper
        :param use_true_agent_models_in_mcts: Whether to use opponent modelling inside of MCTS.
                                            During training: creates a NeuralNetServerHandler
                                            containing the policy for the other agent.
                                            Requires the other agent's model.
                                            During inference: uses opponent modelling head
                                            of self.apprentice.
        :param use_learnt_opponent_models_in_mcts: Wether to learnt opponent models, by querying
                                           the head of apprentice (nn.Module) which is
                                           trained to model opponent actions
                                           (key from prediction dictionary 'policy_0')
        :param action_dim: Shape of actions, use to generate placeholder values
        :param observation_dim: Shape of observations, use to generate placeholder values
        :param num_opponents: Number of opponents that will be playing in an environment
        :param state_preprocess_fn: Function to pre-process observations before they
                                    are fed into the apprentice (an nn.Module)
        :param server_state_preprocess_fn: Same as :param: state_preprocess_fn, but this fn
                                           will be given to underlying NeuralNetServer
        :param use_cuda: Whether to load neural net to a cuda device for action predictions
        '''
        super().__init__(name=name, requires_environment_model=True)
        self.use_cuda = use_cuda
        self.requires_self_prediction = True

        self.algorithm: ExpertIterationAlgorithm = algorithm
        self.expert: Agent = expert
        self.apprentice: nn.Module = apprentice
        if self.use_cuda: self.apprentice = self.apprentice.cuda()

        #### Algorithmic variations ####
        self.use_true_agent_models_in_mcts = use_true_agent_models_in_mcts
        self.use_learnt_opponent_models_in_mcts = use_learnt_opponent_models_in_mcts
        if self.use_true_agent_models_in_mcts and (not self.use_learnt_opponent_models_in_mcts):
            # We will need to set up a server for other agent's models
            # Inside MCTS, we make the evaluation_fn point to the right
            # server (with the other agent's models) on opponent's nodes.
            self.requires_acess_to_other_agents = True
            self.expert.requires_acess_to_other_agents = True

        self.use_agent_modelling: bool = use_agent_modelling
        self.num_opponents: int = num_opponents
        if self.use_agent_modelling:
            self.action_dim = action_dim
            self.observation_dim = observation_dim
            self.requires_opponents_prediction = True
            # Key used to extract opponent policy from extra_info in handle_experienco
            self.extra_info_key = 'probs'  # Allow for 'a' (opponent action) to be used at some point

        # If FALSE, this algorithm is equivalent to DAgger
        self.use_apprentice_in_expert: bool = use_apprentice_in_expert
        if self.use_apprentice_in_expert:
            self.multi_action_requires_server = True
            self.embed_apprentice_in_expert()
        ####

        # Replay buffer style storage
        # Doesn't matter, should not really be using a Storage
        # In tasks with single environments
        self.storage: Storage = self.init_storage(size=100)
        # In vectorized (multiple) environments
        # Mapping from env_id to trajectory storage
        self.storages: Dict[int, Storage] = {}

        # Set state preprocessing functions
        if state_preprocess_fn:
            self.state_preprocess_fn: Callable = state_preprocess_fn
        else:
            self.state_preprocess_fn: Callable = turn_into_single_element_batch
        if server_state_preprocess_fn:
            self.server_state_preprocess_fn: Callable = server_state_preprocess_fn
        else:
            self.server_state_preprocess_fn: Callable = batch_vector_observation

    @Agent.num_actors.setter
    def num_actors(self, n):
        self._num_actors = n
        self.expert.num_actors = n

    @Agent.summary_writer.setter
    def summary_writer(self, summary_writer):
        self._summary_writer = summary_writer
        self.algorithm.summary_writer = summary_writer

    def access_other_agents(self, other_agents_vector: List[Agent], task: 'Task', num_envs):
        '''
        TODO:
        '''
        assert self.use_true_agent_models_in_mcts
        self.expert.access_other_agents(other_agents_vector, task, num_envs)

    def embed_apprentice_in_expert(self):
        # Non-parallel environments
        # TODO: code different variations of brexit for non-parallel envs
        self.expert.policy_fn = self.policy_fn
        self.expert.evaluation_fn = self.evaluation_fn

        # Parallel environments
        if self.use_true_agent_models_in_mcts:
            # Query true opponent model from opponent NeuralNetServerHandler
            self.expert.server_based_policy_fn = \
                self.__class__.opponent_aware_server_based_policy_fn
        elif self.use_learnt_opponent_models_in_mcts:
            # Query learnt opponent model from NeuralNetServerHandler
            self.expert.server_based_policy_fn = \
                self.__class__.learnt_opponent_model_aware_server_based_policy_fn
        else:
            self.expert.server_based_policy_fn = partial(
                request_prediction_from_server,
                key='probs')
        self.expert.server_based_evaluation_fn = partial(
            request_prediction_from_server,
            key='V')

    def init_storage(self, size: int):
        storage = Storage(size=size)
        storage.add_key('normalized_child_visitations')  # \pi_{MCTS} policy
        if self.use_agent_modelling:
            storage.add_key('opponent_s')       # s
            storage.add_key('opponent_policy')  # \pi_{opponent}(.|s)
        return storage

    def handle_experience(self, o, a, r: float, succ_s, done=False,
                          extra_info: Dict[int, Dict[str, Any]] = {}):
        super().handle_experience(o, a, r, succ_s, done)
        expert_child_visitations = self.expert.current_prediction['child_visitations']
        self._handle_experience(o, a, r, succ_s, done, extra_info,
                                self.storage, expert_child_visitations)

    def handle_multiple_experiences(self, experiences: List, env_ids: List[int]):
        # Simplify & explain the idea behind
        # pred_index corresponds to the (tensor) index for self.current_prediction
        # corresponding to environment index (e_i)
        super().handle_multiple_experiences(experiences, env_ids)
        for (o, a, r, succ_o, done, extra_info), (pred_index, e_i) in zip(experiences, enumerate(env_ids)):
            self.storages[e_i] = self.storages.get(e_i, self.init_storage(size=100))
            expert_child_visitations = extra_info['self']['child_visitations']  # self.current_prediction['child_visitations'][pred_index]
            self._handle_experience(o, a, r, succ_o, done, extra_info,
                                    self.storages[e_i], expert_child_visitations)

    def _handle_experience(self, o, a, r, succ_s, done: bool,
                           extra_info: Dict[int, Dict[str, Any]],
                           storage: Storage,
                           expert_child_visitations: List[int]):
        # Preprocessing all variables
        o, r = self.process_environment_signals(o, r)
        normalized_visits = torch.Tensor(self.normalize(expert_child_visitations))

        if self.use_agent_modelling:
            opponent_policy, opponen_obs = self.process_extra_info(extra_info)
        else: opponent_policy, opponen_obs = {}, []

        self.update_storage(storage, o, r, done,
                            opponent_policy=opponent_policy,
                            opponent_s=opponen_obs,
                            mcts_policy=normalized_visits)
        if done: self.handle_end_of_episode(storage)
        self.expert.handle_experience(o, a, r, succ_s, done)

    def handle_end_of_episode(self, storage: Storage):
        self.algorithm.add_episode_trajectory(storage)
        storage.reset()
        if self.algorithm.should_train():
            self.algorithm.train(self.apprentice)
            if self.use_apprentice_in_expert:
                # We need to update the neural net in server used by MCTS
                assert self.expert.server_handler, 'There should be a server'
                self.expert.server_handler.update_neural_net(self.apprentice)

    def process_environment_signals(self, o, r: float):
        processed_s = self.state_preprocess_fn(o)
        processed_r = torch.Tensor([r]).float()
        return processed_s, processed_r

    def process_extra_info(self, extra_info: Dict[str, Any]) \
                           -> Tuple[torch.Tensor, torch.Tensor]:
        # At most there is information about 1 agent
        # Because opponent modelling is only supported
        # For tasks with two agents
        assert len(extra_info) <= 2, ('There can be at most information about 2 agents'
                                      '\'Self\' and 1 other agent')

        if len(extra_info) == 1:  # If dictionary only contains info about this agent
            # First argument to `torch.full` might create an issue (might break for non 1D actions)
            processed_opponent_policy = torch.full((self.action_dim,), float('nan'))
            # Adding batch dimension
            processed_opponent_obs = torch.full(self.observation_dim, float('nan'))
            processed_opponent_obs = self.state_preprocess_fn(processed_opponent_obs)
        else:
            opponent_index = list(filter(lambda key: key != 'self',
                                         extra_info.keys()))[0]  # Not super pretty
            # TODO: extra processing (turn into one hot encoding) will be necessary
            # If using self.extra_info_key = 'a'.
            opponent_policy = extra_info[opponent_index][self.extra_info_key]
            processed_opponent_policy = torch.FloatTensor(opponent_policy)
            processed_opponent_obs = self.state_preprocess_fn(extra_info[opponent_index]['s'])
        return processed_opponent_policy, processed_opponent_obs

    def update_storage(self, storage: Storage,
                       o: torch.Tensor,
                       r: torch.Tensor,
                       done: bool,
                       opponent_policy: torch.Tensor,
                       opponent_s: torch.Tensor,
                       mcts_policy: torch.Tensor):
        storage.add({'normalized_child_visitations': mcts_policy,
                     's': o})
        if self.use_agent_modelling:
            storage.add({'opponent_policy': opponent_policy,
                         'opponent_s': opponent_s})
        if done:
            # Hendrik idea:
            # Using MCTS value for current search might be better?
            for _ in range(len(storage.s)): storage.add({'V': r})

    def normalize(self, x):
        total = sum(x)
        return [x_i / total for x_i in x]

    def model_based_take_action(self, env: Union[gym.Env, List[gym.Env]],
                                observation, player_index: int, multi_action: bool):
        action = self.expert.model_based_take_action(env, observation,
                                                     player_index,
                                                     multi_action)
        self.current_prediction = self.expert.current_prediction
        return action

    def model_free_take_action(self, state, legal_actions: List[int], multi_action: bool = False):
        if self.training: raise RuntimeError('ExpertIterationAgent.model_free_take_action() cannot be called when training is True')
        prediction = self.apprentice(self.state_preprocess_fn(state),
                                     legal_actions=legal_actions)
        return prediction['a']

    def start_server(self, num_connections: int):
        ''' Explain that this is needed because different MCTS experts need to send requests '''
        if num_connections == -1: num_connections = multiprocessing.cpu_count()
        self.expert.server_handler = NeuralNetServerHandler(
            num_connections=num_connections,
            net=self.apprentice,
            preprocess_fn=self.server_state_preprocess_fn
        )

    def close_server(self):
        self.expert.close_server()

    @torch.no_grad()
    def policy_fn(self, observation, legal_actions, self_player_index: int = None, requested_player_index: int = None):
        processed_obs = self.state_preprocess_fn(observation)
        return self.apprentice(processed_obs, legal_actions=legal_actions)['probs'].squeeze(0).numpy()

    @torch.no_grad()
    def evaluation_fn(self, observation, legal_actions):
        processed_obs = self.state_preprocess_fn(observation)
        return self.apprentice(processed_obs, legal_actions=legal_actions)['V'].squeeze(0).numpy()

    def clone(self):
        raise NotImplementedError('Cloning ExpertIterationAgent not supported')

    def __repr__(self):
        agent_stats = (f'Agent modelling: {self.use_agent_modelling}\n'
                       f'Use apprentice in expert: {self.use_apprentice_in_expert}\n'
                       f'Use agent mdelling in mcts: {self.use_true_agent_models_in_mcts}\n'
                       f'Use learnt opponent models in mcts: {self.use_learnt_opponent_models_in_mcts}\n'
                       f'State processing fn: {self.state_preprocess_fn}\n'
                      )
        expert = f"Expert:\n{textwrap.indent(str(self.expert), '    ')}\n"
        algorithm = f"Algorithm:\n{textwrap.indent(str(self.algorithm), '    ')}"
        return agent_stats + expert + algorithm

    #####
    # This is a dirty HACK but oh well...
    @staticmethod
    def opponent_aware_server_based_policy_fn(observation,
                                              legal_actions: List[int],
                                              self_player_index: int,
                                              requested_player_index: int,
                                              connection: Connection,
                                              opponent_connection: Connection) -> np.ndarray:
        key = 'probs'
        target_connection = connection if requested_player_index == self_player_index else opponent_connection
        return request_prediction_from_server(
            observation, legal_actions, target_connection, key)

    @staticmethod
    def learnt_opponent_model_aware_server_based_policy_fn(observation,
                                                           legal_actions: List[int],
                                                           self_player_index: int,
                                                           requested_player_index: int,
                                                           connection: Connection) -> np.ndarray:
        key = 'probs' if requested_player_index == self_player_index else 'policy_0'
        return request_prediction_from_server(
            observation, legal_actions, target_connection, key)
    #
    ####


def choose_feature_extractor(task, config: Dict):
    if config['feature_extractor_arch'] == 'CNN':
        model = Convolutional2DBody(input_shape=config['preprocessed_input_dimensions'],
                                    channels=config['channels'],
                                    kernel_sizes=config['kernel_sizes'],
                                    paddings=config['paddings'],
                                    strides=config['strides'],
                                    residual_connections=config.get('residual_connections', []),
                                    use_batch_normalization=config['use_batch_normalization'])
        return model
    else:
        return ValueError('Only convolutional architectures are supported for ExpertIterationAgent')


def build_apprentice_model(task, config: Dict) -> nn.Module:
    if task.action_type == 'Continuous':
        raise ValueError(f'Only Discrete action type tasks are supported. Task {task.name} has a Continuous action_type')

    feature_extractor = choose_feature_extractor(task, config)

    # REFACTORING: maybe we can refactor into its own function, figure out
    # figure out how to do proper separation of agent modell and not.
    if config['use_agent_modelling']:
        return build_apprentice_with_agent_modelling(
                feature_extractor, task, config)
    else:
        default_embedding_size = [64, 64]
        body = FCBody(
            state_dim=feature_extractor.feature_dim,
            hidden_units=config.get(
                'post_feature_extractor_hidden_units',
                default_embedding_size
            )
        )

        feature_and_body = SequentialBody(feature_extractor, body)

        return CategoricalActorCriticNet(state_dim=feature_and_body.feature_dim,
                                         action_dim=task.action_dim,
                                         critic_gate_fn=config.get('critic_gate_fn', None),
                                         body=feature_and_body)


def build_apprentice_with_agent_modelling(feature_extractor, task, config):
    # TODO: remove hardcodings and place somewhere in config
    # - Embedding size
    # - The bodies are just FC layers (will in the future be RNNs)
    default_embedding_size = 64
    policy_inference_body = FCBody(
        feature_extractor.feature_dim,
        hidden_units=config.get(
            'post_feature_extractor_policy_inference_hidden_units',
            default_embedding_size
        )
    )
    actor_critic_body = FCBody(
        feature_extractor.feature_dim,
        hidden_units=config.get(
            'post_feature_extractor_actor_critic_hidden_units',
            default_embedding_size
        )
    )

    # We model all agents but ourselves
    num_agents_to_model = task.num_agents - 1
    if not isinstance(task.action_dim, (int, float)):
        raise ValueError('number of actions must be an integer (1D)')
    return PolicyInferenceActorCriticNet(feature_extractor=feature_extractor,
                                         num_policies=num_agents_to_model,
                                         num_actions=task.action_dim,
                                         policy_inference_body=policy_inference_body,
                                         actor_critic_body=actor_critic_body)


def build_expert(task, config: Dict, expert_name: str) -> MCTSAgent:
    selection_phase = 'puct' if config['use_apprentice_in_expert'] else 'ucb1'
    exploration = f'exploration_factor_{selection_phase}'
    expert_config = {'budget': config['mcts_budget'],
                     'rollout_budget': config['mcts_rollout_budget'],
                     'selection_phase': selection_phase,
                     'use_dirichlet': config['mcts_use_dirichlet'],
                     exploration: config['mcts_exploration_factor'],
                     'dirichlet_alpha': config['mcts_dirichlet_alpha']}
    return build_MCTS_Agent(task, expert_config, agent_name=expert_name)


def check_parameter_validity(task: 'Task', config: Dict[str, Any]):
    ''' Checks whether :param: config is compatible with :param: task '''
    if config.get('use_agent_modelling', False) and task.num_agents != 2:
        raise NotImplementedError('ExpertIterationAgent with agent modelling '
                                  'is only supported with tasks with 2 agents '
                                  '(one agent is this ExpertIterationAgent and the other '
                                  f'will be the opponent). Given task {task.name} '
                                  f'features {task.num_agents} agents.')
    if (config.get('use_learnt_opponent_models_in_mcts', False) and
        config.get('use_true_agent_models_in_mcts', False)):
        raise ValueError("Both flags 'use_true_agent_models_in_mcts' and "
                         "'use_learnt_opponent_models_in_mcts' were set, which "
                         "is conflicting. One represents "
                         "using true opponent models inside of MCTS, the other "
                         "using learnt opponent models. Read build_ExpertIteration_Agent "
                         "documentation for further info."
                         )


def generate_preprocessing_functions(config) -> Tuple[Callable, Callable]:
    if 'state_preprocessing_fn' in config:
        state_preprocess_fn = parse_preprocessing_fn(
            config['state_preprocessing_fn'])
    else: state_preprocess_fn = None
    if 'server_state_preprocessing_fn' in config:
        server_state_preprocess_fn = parse_preprocessing_fn(
            config['server_state_preprocessing_fn'])
    else: server_state_preprocess_fn = None
    return state_preprocess_fn, server_state_preprocess_fn


def build_ExpertIteration_Agent(task: 'Task',
                                config: Dict[str, Any],
                                agent_name: str = 'ExIt') -> ExpertIterationAgent:
    '''
    TODO: Check all params to make sure they are up to date

    :param task: Environment specific configuration
    :param agent_name: String identifier for the agent
    :param config: Dict contain hyperparameters for the ExpertIterationAgent:
        Higher level params:
        - 'use_apprentice_in_expert': (Bool) whether to bias MCTS's selection
                                      phase and expansion phase with the apprentice.
                                      If False, Expert Iteration becomes the
                                      DAGGER algorithm:
                                      https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf

                                      If True, PUCT will be used as a selection
                                      strategy in MCTS, otherwise UCB1 will be used
        - 'use_agent_modelling': (Bool) Wether to model other agent's actions
                                 as an axuliary task. As in DPIQN paper
        - 'use_true_agent_models_in_mcts': (Bool) Wether to use true agent models
                                           to compute priors for MCTS nodes.
        - 'use_learnt_opponent_models_in_mcts': (Bool) Wether to use learnt agent models
                                                to compute priors for MCTS nodes.

        MCTS params:
        - 'mcts_budget': (Int) Number of iterations of the MCTS loop that will be carried
                                 out before an action is selected.
        - 'mcts_rollout_budget': (Int) Number of steps to simulate during
                                 rollout_phase
        - 'mcts_exploration_factor': (Float) PUCT exploration constant
        - 'mcts_use_dirichlet': (Bool) Whether to add dirichlet noise to the
                                MCTS rootnode's action probabilities (see PUCT)
        - 'mcts_dirichlet_alpha': Parameter of Dirichlet distribution

        (Collected) Dataset params:
        - 'initial_memory_size': (Int) Initial maximum size of replay buffer
        - 'increase_memory_every_n_generations': (Int) Number of iterations to elapse before increasing dataset size.
        - 'increase_memory_size_by': Number of datapoints to increase the size
                                     of the algorithm's dataset everytime the dataset's
                                     size grows, as dictated by
                                     :param: increase_memory_every_n_generations
        - 'end_memory_size': (Int) Ceiling on the size of replay buffer
        - 'num_epochs_per_iteration': (Int) Training epochs to over the game dataset per iteration
        - 'num_games_per_iteration': (Int) Number of episodes to collect before doing a training
        - 'batch_size': (Int) Minibatch size used during training

        Neural Network params:
        - 'learning_rate': (Float) Learning rate for neural network optimizer
        - 'feature_extractor_arch': (str) Architechture for the feature extractor
            + For Convolutional2DBody:
            - 'residual_connections': List[Tuple[int, int]] Which layers should hace residual skip connections
            - 'preprocessed_input_dimensions': Tuple[int] Input dimensions for each channel
            - 'channels': Tuple[int]
            - 'kernel_sizes': Tuple[int]
            - 'paddings': Tuple[int]
        - 'critic_gate_fn': Gating function to be applied to critic's
                            output head. Supported: ['None', 'tanh']
    '''

    check_parameter_validity(task, config)
    apprentice = build_apprentice_model(task, config)
    expert = build_expert(task, config, expert_name=f'Expert:{agent_name}')

    (state_preprocess_fn, server_state_preprocess_fn) = \
        generate_preprocessing_functions(config)

    algorithm = ExpertIterationAlgorithm(
            model_to_train=apprentice,
            batch_size=config['batch_size'],
            num_epochs_per_iteration=config['num_epochs_per_iteration'],
            learning_rate=config['learning_rate'],
            games_per_iteration=config['games_per_iteration'],
            initial_memory_size=config['initial_memory_size'],
            end_memory_size=config['end_memory_size'],
            increase_memory_every_n_generations=config['increase_memory_every_n_generations'],
            increase_memory_size_by=config['increase_memory_size_by'],
            use_agent_modelling=config['use_agent_modelling'],
            num_opponents=(task.num_agents - 1), # We don't model ourselves
            use_cuda=config.get('use_cuda', False)
    )

    return ExpertIterationAgent(
            name=agent_name,
            algorithm=algorithm,
            expert=expert,
            apprentice=apprentice,
            use_apprentice_in_expert=config['use_apprentice_in_expert'],
            use_agent_modelling=config['use_agent_modelling'],
            use_true_agent_models_in_mcts=config['use_true_agent_models_in_mcts'],
            use_learnt_opponent_models_in_mcts=config['use_learnt_opponent_models_in_mcts'],
            action_dim=task.action_dim,
            observation_dim=task.observation_dim,
            num_opponents=(task.num_agents - 1),
            state_preprocess_fn=state_preprocess_fn,
            server_state_preprocess_fn=server_state_preprocess_fn,
            use_cuda=config.get('use_cuda', False)
    )
