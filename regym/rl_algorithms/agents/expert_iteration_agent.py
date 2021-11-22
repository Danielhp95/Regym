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
from regym.networks.utils import parse_gating_fn
from regym.util.data_augmentation import parse_data_augmentation_fn, apply_data_augmentation_to_experiences

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
                 request_observed_action: bool,
                 average_episode_returns_with_mcts_values: bool,
                 action_dim: int,
                 observation_dim: Tuple[int],
                 num_opponents: int,
                 temperature: float=1.,
                 drop_temperature_after_n_moves: int=np.inf,
                 state_preprocess_fn: Optional[Callable]=turn_into_single_element_batch,
                 server_state_preprocess_fn: Optional[Callable]=batch_vector_observation,
                 data_augmnentation_fn: Optional[Callable]=None,
                 use_cuda: bool=False):
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
        :param use_agent_modelling: Whether to model other agent's actions as
                                    an auxiliary task. As in DPIQN paper
        :param use_true_agent_models_in_mcts: Whether to use opponent modelling inside of MCTS.
                                            During training: creates a NeuralNetServerHandler
                                            containing the policy for the other agent.
                                            Requires the other agent's model.
                                            During inference: uses opponent modelling head
                                            of self.apprentice.
        :param use_learnt_opponent_models_in_mcts: Whether to learnt opponent models, by querying
                                           the head of apprentice (nn.Module) which is
                                           trained to model opponent actions
                                           (key from prediction dictionary 'policy_0')
        :param request_observed_action: When using opponent modelling. Whether to request
                                        to store one-hot encoded observed actions.
                                        Otherwise the full distribution over actions is stored
        :param average_episode_returns_with_mcts_values: Whether to average
                                the episode returns with Q values of MCTS' root
                                node, to serve as targets for the apprentice's
                                value head.
        :param action_dim: Shape of actions, use to generate placeholder values
        :param observation_dim: Shape of observations, use to generate placeholder values
        :param num_opponents: Number of opponents that will be playing in an environment
        :param state_preprocess_fn: Function to pre-process observations before they
                                    are fed into the apprentice (an nn.Module)
        :param server_state_preprocess_fn: Same as :param: state_preprocess_fn, but this fn
                                           will be given to underlying NeuralNetServer
        :param data_augmnentation_fn: Function used to augment experiences (create new ones)
                                Currently only implemented for handle_multiple_experiences.
        :param use_cuda: Whether to load neural net to a cuda device for action predictions
        '''
        super().__init__(name=name, requires_environment_model=True)
        self.use_cuda = use_cuda
        self.requires_self_prediction = True

        self.temperature: float = temperature
        self.drop_temperature_after_n_moves: int = drop_temperature_after_n_moves

        self.algorithm: ExpertIterationAlgorithm = algorithm
        self.expert: Agent = expert
        self.apprentice: nn.Module = apprentice
        if self.use_cuda: self.apprentice = self.apprentice.cuda()

        self.average_episode_returns_with_mcts_values = average_episode_returns_with_mcts_values

        #### Algorithmic variations ####
        self._use_true_agent_models_in_mcts = use_true_agent_models_in_mcts
        self._use_learnt_opponent_models_in_mcts = use_learnt_opponent_models_in_mcts
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
            self.request_observed_action = request_observed_action
            self.extra_info_key = 'a' if self.request_observed_action else 'probs'

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
        self.state_preprocess_fn = state_preprocess_fn
        self.server_state_preprocess_fn = server_state_preprocess_fn
        self.data_augmnentation_fn = data_augmnentation_fn

        self.current_episode_lengths = []

    @property
    def use_true_agent_models_in_mcts(self):
        return self._use_true_agent_models_in_mcts

    @property
    def use_learnt_opponent_models_in_mcts(self):
        return self._use_learnt_opponent_models_in_mcts

    @use_learnt_opponent_models_in_mcts.setter
    def use_learnt_opponent_models_in_mcts(self, value: bool):
        '''
        If set, during MCTS search, in order to compute action priors for
        opponent, the learnt opponent model from this agent' apprentice will be queried.
        '''
        if value:
            self._use_true_agent_models_in_mcts = False
            self.requires_acess_to_other_agents = False
            self.expert.requires_acess_to_other_agents = False
        self._use_learnt_opponent_models_in_mcts = value
        self.embed_apprentice_in_expert()

    @use_true_agent_models_in_mcts.setter
    def use_true_agent_models_in_mcts(self, value: bool):
        '''
        If set, during MCTS search, in order to compute action priors for
        opponent, the true opponent model will be queried.
        '''
        if value:
            self._use_learnt_opponent_models_in_mcts = False
        self._use_true_agent_models_in_mcts = value
        self.requires_acess_to_other_agents = value
        self.expert.requires_acess_to_other_agents = value
        self.embed_apprentice_in_expert()

    @Agent.num_actors.setter
    def num_actors(self, n):
        # We would like to just say: super().num_actors = n
        # But python is really annoying when it comes to property setters
        # See: https://bugs.python.org/issue14965
        super(self.__class__, self.__class__).num_actors.fset(self, n)
        self.expert.num_actors = n
        self.current_episode_lengths = [0 for _ in range(self.num_actors)]

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
            # For this agent's nodes: Use the apprentice's actor policy head
            # For opponent nodes: Use true opponent model from opponent NeuralNetServerHandler
            self.expert.server_based_policy_fn = \
                self.__class__.opponent_aware_server_based_policy_fn
        elif self.use_learnt_opponent_models_in_mcts:
            # For this agent's nodes: Use the apprentice's actor policy head
            # For opponent nodes: Use the apprentice's opponent model head
            self.expert.server_based_policy_fn = \
                self.__class__.learnt_opponent_model_aware_server_based_policy_fn
        else:
            # For this agent's nodes: Use the apprentice's actor policy head
            # For opponent nodes: Use the apprentice's actor policy head
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

    def reset_after_episodes(self):
        ''' Resets storages, in case they were half-full between training runs '''
        self.storages: Dict[int, Storage] = {}

    def handle_experience(self, o, a, r: float, succ_s, done=False,
                          extra_info: Dict[int, Dict[str, Any]] = {}):
        super().handle_experience(o, a, r, succ_s, done)
        expert_child_visitations = self.expert.current_prediction['child_visitations']
        expert_state_value_prediction = self.expert.current_prediction['V']
        self._handle_experience(o, a, r, succ_s, done, extra_info,
                                self.storage,
                                expert_child_visitations, expert_state_value_prediction)

    def handle_multiple_experiences(self, experiences: List, env_ids: List[int]):
        super().handle_multiple_experiences(experiences, env_ids)
        if self.data_augmnentation_fn:
            experiences, env_ids = apply_data_augmentation_to_experiences(experiences, env_ids, self.data_augmnentation_fn)
        for (o, a, r, succ_o, done, extra_info), e_i in zip(experiences, env_ids):
            self.storages[e_i] = self.storages.get(e_i, self.init_storage(size=100))
            expert_child_visitations = extra_info['self']['child_visitations']
            expert_state_value_prediction = extra_info['self']['V']
            self._handle_experience(o, a, r, succ_o, done, extra_info,
                                    e_i,
                                    self.storages[e_i],
                                    expert_child_visitations, expert_state_value_prediction)

    def _handle_experience(self, o, a, r, succ_s, done: bool,
                           extra_info: Dict[int, Dict[str, Any]],
                           env_i: int,
                           storage: Storage,
                           expert_child_visitations: torch.FloatTensor,
                           expert_state_value_prediction: torch.FloatTensor):
        self.current_episode_lengths[env_i] += 1
        # Preprocessing all variables
        o, r = self.process_environment_signals(o, r)

        normalized_visits = self._normalize_visitations_with_temperature(
            visitations=expert_child_visitations,
            temperature=(self.temperature if self.current_episode_lengths[env_i] <= self.drop_temperature_after_n_moves else 0.1)
        )

        if self.use_agent_modelling: opponent_policy, opponen_obs = self.process_extra_info(extra_info)
        else: opponent_policy, opponen_obs = {}, []

        self.update_storage(storage, o, r, done,
                            opponent_policy=opponent_policy,
                            opponent_s=opponen_obs,
                            mcts_policy=normalized_visits,
                            expert_state_value_prediction=expert_state_value_prediction)
        if done: self.handle_end_of_episode(storage, env_i)
        self.expert.handle_experience(o, a, r, succ_s, done)

    def _normalize_visitations_with_temperature(self,
                                                visitations: torch.Tensor,
                                                temperature: float):
        '''
        Artificially changing the value of :param: visitations
        via :param: temperature:
            - If temperature > 1: Increases entropy, encouraging exploration
            - If temperature == 1: No changes
            - If temperature < 1: Decreases entropy, encouraging exploitation
        Dropping temperature to 0.1 is equivalent to greedily selecting most visited action
        '''
        expert_child_visitations_with_temperature = visitations ** (1/temperature)
        normalized_visitations = expert_child_visitations_with_temperature / expert_child_visitations_with_temperature.sum()
        return normalized_visitations.clamp(min=0., max=1.)

    def handle_end_of_episode(self, storage: Storage, env_i: int):
        self.current_episode_lengths[env_i] = 0  # We are now beginning a new episode
        self.algorithm.add_episode_trajectory(storage)
        storage.reset()
        if self.algorithm.should_train():
            self.algorithm.train(self.apprentice)
            if self.use_apprentice_in_expert:
                # We need to update the neural net in server used by MCTS
                assert self.expert.server_handler, 'NeuralNetServerHandler missing while trying to update its neural net'
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
            opponent_policy = extra_info[opponent_index][self.extra_info_key]
            processed_opponent_obs = self.state_preprocess_fn(extra_info[opponent_index]['s'])
            if self.extra_info_key == 'a':  # Observing only single actions
                processed_opponent_policy = nn.functional.one_hot(torch.LongTensor([opponent_policy]), num_classes=self.action_dim).squeeze(0)
            elif self.extra_info_key == 'probs':  # Observing full action distribution
                processed_opponent_policy = torch.FloatTensor(opponent_policy)
            else: raise RuntimeError(f'Could not process extra_info_key: {self.extra_info_key}')
        return processed_opponent_policy, processed_opponent_obs

    def update_storage(self, storage: Storage,
                       o: torch.Tensor,
                       r: torch.Tensor,
                       done: bool,
                       opponent_policy: torch.Tensor,
                       opponent_s: torch.Tensor,
                       mcts_policy: torch.Tensor,
                       expert_state_value_prediction: torch.Tensor):
        storage.add({'normalized_child_visitations': mcts_policy,
                     's': o})
        if self.use_agent_modelling:
            storage.add({'opponent_policy': opponent_policy,
                         'opponent_s': opponent_s})
        if self.average_episode_returns_with_mcts_values:
            storage.add({'V': expert_state_value_prediction})
        if done:
            # Hendrik idea:
            # Using MCTS value for current search might be better?
            if self.average_episode_returns_with_mcts_values:
                # Average all previously estimated values with episode return
                storage.V = [(value + r) / 2 for value in storage.V]
            else:
                # Use episodic return for all points
                for _ in range(len(storage.s)): storage.add({'V': r})

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
        basic_stats = f'Name: {self.name}\nRequires access to other agents: {self.requires_acess_to_other_agents}\n'
        agent_stats = (f'Agent modelling: {self.use_agent_modelling}\n'
                       f'Use apprentice in expert: {self.use_apprentice_in_expert}\n'
                       f'Use agent mdelling in mcts: {self.use_true_agent_models_in_mcts}\n'
                       f'Use learnt opponent models in mcts: {self.use_learnt_opponent_models_in_mcts}\n'
                       f'Average episode returns with MCTS values: {self.average_episode_returns_with_mcts_values}\n'
                       f'State processing fn: {self.state_preprocess_fn}\n'
                       f'Server based State processing fn: {self.server_state_preprocess_fn}'
                      )
        agent = f"Agent:\n{textwrap.indent(str(agent_stats), '    ')}\n"
        expert = f"Expert:\n{textwrap.indent(str(self.expert), '    ')}\n"
        algorithm = f"Algorithm:\n{textwrap.indent(str(self.algorithm), '    ')}"
        return basic_stats + agent + expert + algorithm

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
            observation, legal_actions, connection, key)
    #
    ####


def choose_feature_extractor(task, config: Dict):
    if config['feature_extractor_arch'] == 'CNN':
        model = Convolutional2DBody(input_shape=config['preprocessed_input_dimensions'],
                                    channels=config['channels'],
                                    kernel_sizes=config['kernel_sizes'],
                                    paddings=config['paddings'],
                                    strides=config['strides'],
                                    final_feature_dim=config['final_feature_dim'],
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
        return build_apprentice_no_agent_modelling(feature_extractor, config, task)

def build_apprentice_no_agent_modelling(feature_extractor, config, task) -> nn.Module:
    default_embedding_size = [64, 64]
    body = FCBody(
        state_dim=feature_extractor.feature_dim,
        hidden_units=config.get(
            'post_feature_extractor_hidden_units',
            default_embedding_size
        )
    )

    feature_and_body = SequentialBody([feature_extractor, body])

    critic_gate_fn = parse_gating_fn(config.get('critic_gate_fn', None))

    return CategoricalActorCriticNet(state_dim=feature_and_body.feature_dim,
                                     action_dim=task.action_dim,
                                     critic_gate_fn=critic_gate_fn,
                                     body=feature_and_body)


def build_apprentice_with_agent_modelling(feature_extractor, task, config):
    default_embedding_size = [64]
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
    expert_config = {
        'budget': config['mcts_budget'],
        'rollout_budget': config.get('mcts_rollout_budget', 0.),
        'selection_phase': selection_phase,
        'use_dirichlet': config.get('mcts_use_dirichlet', False),
        exploration: config['mcts_exploration_factor'],
        'dirichlet_alpha': config['mcts_dirichlet_alpha'],
        'dirichlet_strength': config.get('mcts_dirichlet_strength', 1.)
    }
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
        - 'use_agent_modelling': (Bool) Whether to model other agent's actions
                                 as an axuliary task. As in DPIQN paper
        - 'use_true_agent_models_in_mcts': (Bool) Whether to use true agent models
                                           to compute priors for MCTS nodes.
        - 'use_learnt_opponent_models_in_mcts': (Bool) Whether to use learnt agent models
                                                to compute priors for MCTS nodes.
        - 'request_observed_action': Whether to observe one hot encoded actions, otherwise full policy will be requested.
                                     Only meaningful when :param: use_agent_modelling is set.
        - 'average_episode_returns_with_mcts_values': (Bool) Whether to average
                                the episode returns with Q values of MCTS' root
                                node, to serve as targets for the apprentice's
                                value head. Idea taken from: https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628

        MCTS params:
        - 'mcts_budget': (Int) Number of iterations of the MCTS loop that will be carried
                                 out before an action is selected.
        - 'mcts_rollout_budget': (Int) Number of steps to simulate during
                                 rollout_phase
        - 'mcts_exploration_factor': (Float) PUCT exploration constant
        - 'mcts_use_dirichlet': (Bool) Whether to add dirichlet noise to the
                                MCTS rootnode's action probabilities (see PUCT)
        - 'mcts_dirichlet_alpha': Parameter of Dirichlet distribution
        - 'temperature': Value by which MCTS child visitations will be
                         inversely exponentiated to (N^(1/temperature))
        - 'drop_temperature_after_n_moves': Number of moves after which
                                            temperature parameter will dropped
                                            to a very small value (around 0.01)

        (Collected) Dataset params:
        - 'initial_max_generations_in_memory': (Int) Initial number of generations to be allowed
                                               in replay buffer
        - 'increase_memory_every_n_generations': (Int) Number of iterations to elapse before increasing dataset size.
        - 'memory_increase_step': Number of extra generations to allow in the
                                  algorithm's dataset everytime the dataset's
                                  capacity increases, as dictated by
                                  :param: increase_memory_every_n_generations
        - 'final_max_generations_in_memory': (Int) Ceiling on the size of replay buffer
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
            - 'final_feature_dim': int. Dimensionality of the final, fully connected layer of a convolutional body
        - 'critic_gate_fn': Gating function to be applied to critic's
                            output head. Supported: ['None', 'tanh']
    '''

    check_parameter_validity(task, config)
    apprentice = build_apprentice_model(task, config)
    expert = build_expert(task, config, expert_name=f'Expert:{agent_name}')

    (state_preprocess_fn, server_state_preprocess_fn) = \
        generate_preprocessing_functions(config)

    data_augmnentation_fn = parse_data_augmentation_fn(config['data_augmnentation_fn']) \
        if 'data_augmnentation_fn' in config else None

    algorithm = ExpertIterationAlgorithm(
            model_to_train=apprentice,
            batch_size=config['batch_size'],
            num_epochs_per_iteration=config['num_epochs_per_iteration'],
            learning_rate=config['learning_rate'],
            games_per_iteration=config['games_per_iteration'],
            initial_max_generations_in_memory=config['initial_max_generations_in_memory'],
            final_max_generations_in_memory=config['final_max_generations_in_memory'],
            increase_memory_every_n_generations=config['increase_memory_every_n_generations'],
            memory_increase_step=config['memory_increase_step'],
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
            request_observed_action=config.get('request_observed_action', False),
            average_episode_returns_with_mcts_values=config.get('average_episode_returns_with_mcts_values', False),
            action_dim=task.action_dim,
            observation_dim=task.observation_dim,
            num_opponents=(task.num_agents - 1),
            state_preprocess_fn=state_preprocess_fn,
            server_state_preprocess_fn=server_state_preprocess_fn,
            use_cuda=config.get('use_cuda', False),
            temperature=config.get('temperature', 1.),
            drop_temperature_after_n_moves=config.get('drop_temperature_after_n_moves', np.inf),
            data_augmnentation_fn=data_augmnentation_fn
    )
