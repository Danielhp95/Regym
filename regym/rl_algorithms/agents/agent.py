from typing import List, Union, Any
from abc import ABC, abstractmethod

import gym


class Agent(ABC):
    '''
    This Agent class is an abstract interface for all other Agent subclasses.

    An agent is an entity that communicates with an environment for 2 purposes:
        - To take actions in said environment using a policy.
        - To collect experiences with which to feed to an underlying algorithm
          that will update the Agent's policy. An experience is a Tuple of:
            - (state, action, reward, successor state, done)

    Each algorithm (PPO, DQN, REINFORCE, ExpertIteration) can heavily differ:
        - How actions are taken: using a tabular method, neural networks,
          an environment model...
        - How the environment experiences are handled
          (i.e. when they are removed from memory, which experiences to keep)

    Agents can be model-based or model-free. By default agents are model-free,
    as specified by their property `Agent.requires_environment_model = False`.
    The difference between model-based and model-free agents is that the former
    receive a copy of the environment every time they take an action so that
    they can perform search on them. Model-free agents instead receive only
    a copy of their respective environment observation.

    Agents should be built using their corresponding build_X_Agent() function
    exposed in `regym.rl_algorithms`. Only craete an agent using their __init__
    function if you know what you are doing.
    '''

    def __init__(self, name: str, requires_environment_model: bool = False):
        '''
        By default agents do not require an environment model.
        What this means is that agents take actions based on environment
        observations (although agents may internally store more information).
        Agents which do depend on an environment model will receive a copy
        of the environment every time they are asked to return an action.

        :param name: String identifier, Agents are named,
                     for all great creations should be named
        :param requires_environment_model: Flag to signal whether the agent
                                           will receive a copy of the
                                           environment at each decision point
        '''
        self.requires_environment_model: bool = requires_environment_model
        self.name: str = name
        self.training: bool = True
        self.handled_experiences: int = 0

    def model_based_take_action(self, env: Union[gym.Env, List[gym.Env]],
                                legal_actions: Union[List[int], List[List[int]]],
                                multi_action: bool):
        '''
        This function is called inside of an `regym.rl_loops`, asking
        this Agent to take an action at a given state in the environment
        where :param: observation is observed, so that the environment
        may move forward.

        This function will only be called by an regym.rl_loop if this agent's
        `self.requires_environment_model` flag is set.

        :param env: A deepcopy of the environment where this agent is acting
        :param observation: Observation at current timestep for this agent.
                            This is necessary because observations cannot be
                            accessed directly from OpenAI gym :param: env.
        :param player_index: Index of this agent in the environment's
                             agent vector.
        :param multi_action: Whether to consider :param: observation
                             and :param: env as a vector of observations /
                             environments
        :returns: Action to be executed on the environment
        '''
        raise NotImplementedError('To be implemented in Subclass')

    def model_free_take_action(self, observation: Union[Any, List[Any]],
                               legal_actions: Union[List[int], List[List[int]]],
                               multi_action: bool):
        '''
        This function is called inside of an regym.rl_loops, asking
        this Agent to take an action at a given state in the environment
        so that the environment model may move forward.

        :param observation: Observation at current timestep for this agent.
                            This is necessary because observations cannot be
                            accessed directly from OpenAI gym :param: env.
        :param legal_actions: List of actions available to agent at
                              :param: observation
        :param multi_action: Whether to consider :param: observation as
                             a list of observations, each corresponding
                             to the observation of a different environment
        :returns: Action(s) to be executed on the environment
        '''
        raise NotImplementedError('To be implemented in Subclass')

    @abstractmethod
    def handle_experience(self, o, a, r, succ_o, done=False):
        '''
        Processes a single 'experience' (defined by the parameters of this function),
        which is the main method of gathering data of an RL algorithm.
        NOTE: Unless this agent's 'training' flag is set to True,
        this function will not be called.

        :param o:      Environment observation
        :param a:      action taken by this agent at :param s: observation
        :param r:      reward obtained by this agent after taking
                       :param a: action at :param o: observation
        :param succ_o: Environment observation reached after after taking
                       :param a: action at :param o: observation
        :param done:   Boolean denoting episode termination
        '''
        self.handled_experiences += 1

    def handle_multiple_experiences(self, experiences: List, env_ids: List[int]):
        '''
        TODO
        '''
        self.handled_experiences += len(experiences)

    @abstractmethod
    def clone(self):
        '''
        Function which should return an identical copy of the agent
        INCLUDING a deep copy of all the underlying objects used by the agent
        such as neural networks, storage, agent flags... etc
        '''
        raise NotImplementedError('To be implemented in Subclass')
