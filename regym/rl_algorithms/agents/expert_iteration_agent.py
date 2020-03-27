from typing import List, Dict

import gym
import torch.nn as nn

from regym.rl_algorithms.replay_buffers import Storage
from regym.rl_algorithms.networks import Convolutional2DBody, FCBody, CategoricalActorCriticNet, SequentialBody

from regym.rl_algorithms.agents import Agent, build_MCTS_Agent, MCTSAgent
from regym.rl_algorithms.agents import Agent, MCTSAgent


class ExpertIterationAgent(Agent):

    def __init__(self, name: str,
                 expert: MCTSAgent, apprentice: nn.Module,
                 memory_size: int,
                 use_agent_modelling: bool = False  # TODO: remove default val
                 ):
        '''
        :param name: String identifier for the agent
        :param expert: Agent used to take actions in the environment
                       and create optimization targets for the apprentice
        :param memory_size: size of "replay buffer"
        :param apprentice: TODO
        '''
        super().__init__(name=name, requires_environment_model=True)
        self.expert = expert
        self.apprentice = apprentice
        self.use_agent_modelling = use_agent_modelling

        self.current_prediction: Dict = {}
        self.storage = self.init_storage(size=memory_size)

    def init_storage(self, size: int):
        storage = Storage(size=size)
        storage.add_key('normalized_child_visitations')
        return storage

    def handle_experience(self, s, a, r, succ_s, done=False):
        super().handle_experience(s, a, r, succ_s, done)

        normalized_visits = self.normalize(
                self.expert.current_prediction['child_visitations'])

        # TODO: get opponents action from current_prediction

        self.storage.add({'normalized_child_visitations': normalized_visits,
                          's': s})


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
        - 'memory_size': (Int) size of "replay buffer"

        MCTS params:
        - 'mcts_budget': (Int) Number of iterations of the MCTS loop that will be carried
                                 out before an action is selected.
        - 'mcts_rollout_budget': (Int) Number of steps to simulate during
                                 rollout_phase
TODO    - 'use_agent_modelling: (Bool) whether to model agent's policies as in DPIQN paper

        Neural Network params:
TODO    - 'batch_size': (Int) Minibatch size used during training
        - 'feature_extractor_arch': (str) Architechture for the feature extractor
            + For Convolutional2DBody:
            - 'preprocessed_input_dimensions': Tuple[int] Input dimensions for each channel
            - 'channels': Tuple[int]
            - 'kernel_sizes': Tuple[int]
            - 'paddings': Tuple[int]
    '''

    apprentice = build_apprentice_model(task, config)
    expert = build_expert(task, config, expert_name=f'Expert:{agent_name}')
    return ExpertIterationAgent(name=agent_name,
                                expert=expert,
                                apprentice=apprentice,
                                memory_size=config['memory_size'],
                                use_agent_modelling=bool(config['use_agent_modelling']))
