from typing import List, Dict

import gym
import torch
import torch.nn as nn

from regym.rl_algorithms.replay_buffers import Storage
from regym.rl_algorithms.networks import Convolutional2DBody, FCBody, CategoricalActorCriticNet, SequentialBody

from regym.rl_algorithms.agents import Agent, build_MCTS_Agent, MCTSAgent
from regym.rl_algorithms.agents import Agent, MCTSAgent

from regym.rl_algorithms.expert_iteration import ExpertIterationAlgorithm


class ExpertIterationAgent(Agent):

    def __init__(self, algorithm: ExpertIterationAlgorithm,
                 name: str,
                 expert: MCTSAgent, apprentice: nn.Module,
                 use_apprentice_in_expert: bool,
                 memory_size: int,
                 use_agent_modelling: bool
                 ):
        '''
        :param algorithm: TODO
        :param name: String identifier for the agent
        :param expert: Agent used to take actions in the environment
                       and create optimization targets for the apprentice
        :param apprentice: TODO
        :param use_agent_modelling: TODO
        :param memory_size: Max number of datapoints to be stored from game examples. TODO: not passing memory here?
        :param use_apprentice_in_expert: whether to bias MCTS's selection
                                         phase and expansion phase with the
                                         apprentice.
        :param memory_size: size of "replay buffer"
        '''
        super().__init__(name=name, requires_environment_model=True)
        self.algorithm = algorithm
        self.expert = expert
        self.apprentice = apprentice

        #### Algorithmic variations  ####
        self.use_apprentice_in_expert = use_apprentice_in_expert  # If FALSE, this algorithm is equivalent to DAgger
        self.use_agent_modelling = use_agent_modelling

        self.current_prediction: Dict = {}

        # Replay buffer style storage
        self.storage = self.init_storage(size=memory_size)
        self.current_episode_length = 0

        self.state_preprocess_function = self.PRE_PROCESSING

    def PRE_PROCESSING(self, x):
        '''
        Required to save model, as it was previously in lambda function
        '''
        return torch.from_numpy(x).unsqueeze(0).type(torch.FloatTensor)

    def init_storage(self, size: int):
        storage = Storage(size=size)
        storage.add_key('normalized_child_visitations')
        # TODO: it might be necessary to add a `legal_actions` key
        return storage

    def handle_experience(self, s, a, r: float, succ_s, done=False):
        super().handle_experience(s, a, r, succ_s, done)
        self.current_episode_length += 1

        s, r = self.process_environment_signals(s, r)

        normalized_visits = self.normalize(self.expert.current_prediction['child_visitations'])

        self.update_storage(s, r, done,
                            mcts_policy=normalized_visits)

        if done:
            self.current_episode_length = 0
            self.algorithm.train(self.apprentice,
                                 dataset=self.storage)

    def process_environment_signals(self, s, r: float):
        processed_s = self.state_preprocess_function(s)
        processed_r = torch.Tensor([r]).float()
        return processed_s, processed_r

    def update_storage(self, s: torch.Tensor, r: torch.Tensor, done: bool,
                       mcts_policy: torch.Tensor):
        # TODO: get opponents action from current_prediction
        self.storage.add({'normalized_child_visitations': mcts_policy,
                          's': s})
        if done:
            for _ in range(self.current_episode_length):
                self.storage.add({'v': r})
        # TODO: If storage gets too big, remove it? 

    def normalize(self, x):
        total = sum(x)
        return [x_i / total for x_i in x]

    def take_action(self, env: gym.Env, player_index: int):
        action = self.expert.take_action(env, player_index)
        return action

    def clone(self):
        raise NotImplementedError('Cloning ExpertIterationAgent not supported')


def choose_feature_extractor(task, config: Dict):
    # TODO: Mess around with connect4 and conv bodies to see what to do.
    if config['feature_extractor_arch'] == 'CNN':
        model = Convolutional2DBody(input_shape=config['preprocessed_input_dimensions'],
                                    channels=config['channels'],
                                    kernel_sizes=config['kernel_sizes'],
                                    paddings=config['paddings'],
                                    strides=config['strides'])
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
                                     phi_body=feature_and_body)

def build_expert(task, config: Dict, expert_name: str) -> MCTSAgent:
    expert_config = {'budget': config['mcts_budget'],
                     'rollout_budget': config['mcts_rollout_budget']}
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


        MCTS params:
        - 'mcts_budget': (Int) Number of iterations of the MCTS loop that will be carried
                                 out before an action is selected.
        - 'mcts_rollout_budget': (Int) Number of steps to simulate during
                                 rollout_phase
TODO    - 'use_agent_modelling: (Bool) whether to model agent's policies as in DPIQN paper

        (Collected) Dataset params:
        - 'initial_memory_size': (Int) Initial maximum size of replay buffer
        - 'memory_size_increase_frequency': (Int) Number of iterations to elapse before increasing dataset size.
TODO    - 'end_memory_size': (Int) Ceiling on the size of replay buffer
        - 'num_epochs_per_iteration': (Int) Training epochs to over the game dataset per iteration
TODO    - 'num_games_per_iteration': (Int) Number of episodes to collect before doing a training
        - 'batch_size': (Int) Minibatch size used during training

        Neural Network params:
        - 'learning_rate': (Float) Learning rate for neural network optimizer
        - 'feature_extractor_arch': (str) Architechture for the feature extractor
            + For Convolutional2DBody:
            - 'preprocessed_input_dimensions': Tuple[int] Input dimensions for each channel
            - 'channels': Tuple[int]
            - 'kernel_sizes': Tuple[int]
            - 'paddings': Tuple[int]
    '''

    apprentice = build_apprentice_model(task, config)
    expert = build_expert(task, config, expert_name=f'Expert:{agent_name}')

    algorithm = ExpertIterationAlgorithm(model_to_train=apprentice,
                                         batch_size=config['batch_size'],
                                         num_epochs_per_iteration=config['num_epochs_per_iteration'],
                                         learning_rate=config['learning_rate'],
                                         memory_size_increase_frequency=config['memory_size_increase_frequency'],
                                         initial_memory_size=config['initial_memory_size'],
                                         end_memory_size=config['end_memory_size'])

    return ExpertIterationAgent(algorithm=algorithm,
                                name=agent_name,
                                expert=expert,
                                apprentice=apprentice,
                                memory_size=config['initial_memory_size'],  # TODO: remove this from here
                                use_apprentice_in_expert=config['use_apprentice_in_expert'],
                                use_agent_modelling=bool(config['use_agent_modelling']))
