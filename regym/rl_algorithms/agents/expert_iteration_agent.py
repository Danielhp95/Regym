from typing import List, Dict, Any, Callable, Union
import multiprocessing
import textwrap

import gym
import torch
import torch.nn as nn

from regym.rl_algorithms.replay_buffers import Storage

from regym.rl_algorithms.networks import Convolutional2DBody, FCBody, CategoricalActorCriticNet, SequentialBody
from regym.rl_algorithms.networks.preprocessing import turn_into_single_element_batch

from regym.rl_algorithms.agents import Agent, build_MCTS_Agent, MCTSAgent

from regym.rl_algorithms.expert_iteration import ExpertIterationAlgorithm

from regym.rl_algorithms.servers.neural_net_server import NeuralNetServerHandler


class ExpertIterationAgent(Agent):

    def __init__(self, algorithm: ExpertIterationAlgorithm,
                 name: str,
                 expert: MCTSAgent, apprentice: nn.Module,
                 use_apprentice_in_expert: bool,
                 use_agent_modelling: bool):
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
                                    an axuliary task. As in DPIQN paper
        '''
        super().__init__(name=name, requires_environment_model=True)
        self.algorithm: ExpertIterationAlgorithm = algorithm
        self.expert: Agent = expert
        self.apprentice: nn.Module = apprentice

        #self.fake_expert = expert.clone()
        #self.fake_expert.policy_fn = self.expert.random_selection_policy
        #self.fake_expert.use_dirichlet = False

        #### Algorithmic variations ####
        self.use_apprentice_in_expert: bool = use_apprentice_in_expert  # If FALSE, this algorithm is equivalent to DAgger
        if use_apprentice_in_expert:
            self.multi_action_requires_server = True
            self.embed_apprentice_in_expert()

        self.use_agent_modelling: bool = use_agent_modelling
        ####

        # Replay buffer style storage
        # Doesn't matter, should not really be using a Storage
        # In tasks with single environments
        self.storage: Storage = self.init_storage(size=100)
        # In vectorized (multiple) environments
        # Mapping from env_id to trajectory storage
        self.storages: Dict[int, Storage] = {}

        self.state_preprocess_fn: Callable = turn_into_single_element_batch

    def embed_apprentice_in_expert(self):
        # Non-parallel environments
        self.expert.policy_fn = self.policy_fn
        self.expert.evaluation_fn = self.evaluation_fn
        # Parallel environments
        self.expert.server_based_policy_fn = self.__class__.server_based_policy_fn
        self.expert.server_based_evaluation_fn = self.__class__.server_based_evaluation_fn
        
    def init_storage(self, size: int):
        storage = Storage(size=size)
        storage.add_key('normalized_child_visitations')
        return storage

    def handle_experience(self, o, a, r: float, succ_s, done=False):
        super().handle_experience(o, a, r, succ_s, done)
        o, r = self.process_environment_signals(o, r)
        normalized_visits = torch.Tensor(self.normalize(self.expert.current_prediction['child_visitations']))
        self.update_storage(self.storage, o, r, done,
                            mcts_policy=normalized_visits)
        if done: self.handle_end_of_episode(self.storage)
        self.expert.handle_experience(o, a, r, succ_s, done)

    def handle_multiple_experiences(self, experiences: List, env_ids: List[int]):
        for (o, a, r, succ_o, done), e_i in zip(experiences, env_ids):
            self.storages[e_i] = self.storages.get(e_i, self.init_storage(size=100))
            o_prime, r_prime = self.process_environment_signals(o, r)
            normalized_visits = torch.Tensor(self.normalize(self.expert.current_prediction[e_i]['child_visitations']))
            self.update_storage(self.storages[e_i], o_prime, r_prime, done,
                                normalized_visits)
            if done:
                # Check that by deleting you don't remove datapoints for self.algorithm
                self.handle_end_of_episode(self.storages[e_i])
                del self.storages[e_i]

    def handle_end_of_episode(self, storage: Storage):
        self.algorithm.add_episode_trajectory(storage)
        storage.reset()
        if self.algorithm.should_train():
            self.algorithm.train(self.apprentice)
            self.server_handler.update_neural_net(self.apprentice)

    def process_environment_signals(self, o, r: float):
        processed_s = self.state_preprocess_fn(o)
        processed_r = torch.Tensor([r]).float()
        return processed_s, processed_r

    def update_storage(self, storage: Storage, o: torch.Tensor,
                       r: torch.Tensor, done: bool,
                       mcts_policy: torch.Tensor):
        # TODO: get opponents action from current_prediction
        storage.add({'normalized_child_visitations': mcts_policy, 's': o})
        if done:
            # Hendrik idea:
            # Using MCTS value for current search might be better?
            for _ in range(len(storage.s)): storage.add({'v': r})

    def normalize(self, x):
        total = sum(x)
        return [x_i / total for x_i in x]

    def model_based_take_action(self, env: Union[gym.Env, List[gym.Env]],
                                observation, player_index: int, multi_action: bool):
        action = self.expert.model_based_take_action(env, observation,
                                                     player_index,
                                                     multi_action)
        #fake_action = self.fake_expert.model_based_take_action(env, observation, player_index)
        #fake_pi_mcts = self.normalize(self.fake_expert.current_prediction['child_visitations'])
        #distance_vector = [abs(pi_a_mcts - pi_a_nn)
        #                   for pi_a_nn, pi_a_mcts
        #                   in zip(self.policy_fn(observation, env.get_moves()), fake_pi_mcts)]
        return action

    def model_free_take_action(self, state, legal_actions: List[int], multi_action: bool = False):
        if self.training: raise RuntimeError('ExpertIterationAgent.model_free_take_action() cannot be called when training is True')
        prediction = self.apprentice(self.state_preprocess_fn(state),
                                     legal_actions=legal_actions)
        return prediction['a']

    def start_server(self, num_connections):
        ''' Explain that this is needed because different MCTS experts need to send requests '''
        self.server_handler = NeuralNetServerHandler(
            num_connections=num_connections, net=self.apprentice)
        self.expert.server_handler = self.server_handler
        del self.apprentice

    @torch.no_grad()
    def policy_fn(self, observation, legal_actions):
        processed_obs = self.state_preprocess_fn(observation)
        return self.apprentice(processed_obs, legal_actions=legal_actions)['probs'].squeeze(0).numpy()

    @torch.no_grad()
    def evaluation_fn(self, observation, legal_actions):
        processed_obs = self.state_preprocess_fn(observation)
        return self.apprentice(processed_obs, legal_actions=legal_actions)['v'].squeeze(0).numpy()

    def clone(self):
        raise NotImplementedError('Cloning ExpertIterationAgent not supported')

    def __repr__(self):
        agent_stats = f'Agent modelling: {self.use_agent_modelling}\nUse apprentice in expert: {self.use_apprentice_in_expert}\n'
        expert = f"Expert:\n{textwrap.indent(str(self.expert), '    ')}\n"
        algorithm = f"Algorithm:\n{textwrap.indent(str(self.algorithm), '    ')}"
        return agent_stats + expert + algorithm

    #####
    # These two methods are a dirty hack!
    @staticmethod
    def server_based_policy_fn(observation, legal_actions, connection):
        connection.send((observation, legal_actions))
        prediction = connection.recv()
        return prediction['probs'].squeeze(0).numpy()


    @staticmethod
    def server_based_evaluation_fn(observation, legal_actions, connection):
        connection.send((observation, legal_actions))
        prediction = connection.recv()
        return prediction['v'].squeeze(0).numpy()
    #
    #####


def choose_feature_extractor(task, config: Dict):
    # TODO: Mess around with connect4 and conv bodies to see what to do.
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

    feature_extractor_body = choose_feature_extractor(task, config)

    # REFACTORING: maybe we can refactor into its own function, figure out
    # figure out how to do proper separation of agent modell and not.
    if config['use_agent_modelling']:
        raise ValueError('Agent modelling not yet supported')
    else:
        body = FCBody(state_dim=feature_extractor_body.feature_dim,
                      hidden_units=(64, 64))

    feature_and_body = SequentialBody(feature_extractor_body, body)

    return CategoricalActorCriticNet(state_dim=feature_and_body.feature_dim,
                                     action_dim=task.action_dim,
                                     critic_gate_fn=config.get('critic_gate_fn', None),
                                     phi_body=feature_and_body)

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


def build_ExpertIteration_Agent(task, config, agent_name):
    '''
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
        - 'memory_size_increase_frequency': (Int) Number of iterations to elapse before increasing dataset size.
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

    apprentice = build_apprentice_model(task, config)
    expert = build_expert(task, config, expert_name=f'Expert:{agent_name}')

    algorithm = ExpertIterationAlgorithm(
            model_to_train=apprentice,
            batch_size=config['batch_size'],
            num_epochs_per_iteration=config['num_epochs_per_iteration'],
            learning_rate=config['learning_rate'],
            games_per_iteration=config['games_per_iteration'],
            memory_size_increase_frequency=config['memory_size_increase_frequency'],
            initial_memory_size=config['initial_memory_size'],
            end_memory_size=config['end_memory_size'])

    return ExpertIterationAgent(
            name=agent_name,
            algorithm=algorithm,
            expert=expert,
            apprentice=apprentice,
            use_apprentice_in_expert=config['use_apprentice_in_expert'],
            use_agent_modelling=config['use_agent_modelling'])
