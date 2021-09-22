from typing import List, Dict, Union, Any, Callable
from abc import ABC, abstractmethod

import gym
import torch

from regym.networks.servers import neural_net_server


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

    def __init__(self, name: str, requires_environment_model: bool = False,
                 multi_action_requires_server: bool = False):
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
        :param multi_action_requires_server: Flag to signal whether the agent
                                             needs to spawn a
                                             NeuralNetServerHandler in parallel
                                             environments (RegymAsyncVectorEnv)
        '''
        self.name: str = name
        self.training: bool = True

        self.handled_experiences: int = 0
        self.finished_episodes: int = 0

        self.requires_environment_model: bool = requires_environment_model
        self.multi_action_requires_server: bool = multi_action_requires_server

        self._num_actors: int = 1
        self.server_handler: NeuralNetServerHandler = None

        '''
        Dictionary containing agent-specific information regarding the last
        action that was taken.
        (Like values computed during a forward pass of a neural network)
        '''
        self.current_prediction: Dict[str, Any] = None

        '''
        Flag denoting if this agent requires information
        from other opponents as part of any of the following:
            - Compute an action
            - Update its policy
            - Make new friends
        '''
        self.requires_opponents_prediction: bool = False
        '''
        Flag denoting whether an agent needs access to past predictions.
        Useful in multienv scenarios where an internal agent state might be
        overriden with new predictions information required once the
        `handle_experience` funciton is called.
        Look into ExpertIterationAgent for an example.
        '''
        self.requires_self_prediction: bool = False
        '''
        Flag denoting whether an agent needs access
        to the other agents in an environment prior to
        episodes being run

        TODO: agents that have this flag should be given
        access to other agents as soon as they do a model update
        '''
        self.requires_acess_to_other_agents: bool = False

        self._state_preprocess_fn = self.identity_fn

        # Keys from self.__dict__ that will be ignored when pickling
        # an agent. Each agent subclass can incorporate new keys.
        # Read into object.__getstate__ for more info!
        self.keys_to_not_pickle = ['server_handler', '_summary_writer']

    @property
    def num_actors(self) -> int:
        '''
        Property denoting how many individual actors this agent represents
        (i.e on how many environments it is acting)
        '''
        return self._num_actors

    @num_actors.setter
    def num_actors(self, n: int):
        '''
        This setter can be overriden by subclasses, as each Agent / algorithm
        can feature different logic to deal with multiple actors. Such as:
        - PPO requiring a unique storage for each actor
        - Expert iteration requires a NeuralNetServerHandler with a number of
          connections equal to this `num_actors` property
        - DQN only needs a single experience replay, regardless of `num_actors`
        '''
        self._num_actors = n

    @property
    def summary_writer(self) -> torch.utils.tensorboard.SummaryWriter:
        '''
        torch.util.tensorboard.SummaryWritter. Super useful for logging
        anything from performance (like loss computation) or timing (time that
        it takes per training call)
        '''
        return self._summary_writer

    @summary_writer.setter
    def summary_writer(self, summary_writer: torch.utils.tensorboard.SummaryWriter):
        self._summary_writer = summary_writer

    @property
    def state_preprocess_fn(self) -> Callable[[Any], torch.Tensor]:
        '''
        Function used to process environment observations. Normal usecases
        include preprocessing environment observations before a prediction
        is computed to take an action (during `Agent.model_free_take_action()`)
        or before inserting an observation in an algorithm specific dataset.
        '''
        return self._state_preprocess_fn

    # Reasonable default for state_preprocess_fn. We would use lambda x: x,
    # but Pickle isn't a big friend of lambda functions
    def identity_fn(self, x): return torch.tensor(x)

    @state_preprocess_fn.setter
    def state_preprocess_fn(self, state_preprocess_fn: Callable):
        if not isinstance(state_preprocess_fn, Callable):
            error_msg = f'Attempted to set a {type(state_preprocess_fn)} as a state_preprocess_fn, it must be a Callable'
            raise ValueError(error_msg)
        self._state_preprocess_fn = state_preprocess_fn

    def model_based_take_action(self, env: Union[gym.Env, List[gym.Env]],
                                legal_actions: Union[List[int], List[List[int]]],
                                player_index: int, multi_action: bool):
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

    def handle_experience(self, o, a, r, succ_o, done=False,
                          external_agent_infos={}):
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
        :param external_agent_infos: Information about other agents
                                     current_prediction (described in Agent.__init__)
        '''
        self.handled_experiences += 1
        if done: self.finished_episodes += 1

    def handle_multiple_experiences(self, experiences: List, env_ids: List[int]):
        '''
        The multiactor variant of function `Agent.handle_experience()`.

        :param env_ids: is important for multiagent _sequential_ games, because
        not all agents act on every single environment state, and thus for a
        multiple actor (parallel), multiagent environment with `N` parallel
        environments and `M` agents, for any given timestep, an arbitrary
        subset of the latest experiences (n <= N) might go to to each of
        the agent (a_i for 0 < i < M). In simultaneous games and single agent
        environments :param env_ids: is irrelevant.

        :param experiences: List of Tuples of format (o, a, r, succ_o, done) as
                            defined in the docs of `Agent.handle_experience()`.
        :param env_ids:     Indeces of the environments where :param: experiences
                            come from.
        '''
        self.handled_experiences += len(experiences)
        for (_, _, _, _, done, _) in experiences:
            if done: self.finished_episodes += 1

    def access_other_agents(self,
                            other_agents_vector: List['Agent'],
                            task: 'Task',
                            num_envs: int):
        '''
        Function that grants this agent access to all :param: other_agents_vector,
        which act on :param: task BEFORE simulation. Useful, for instance,
        if an agent requires knowledge about other agent's internals

        Will only be invoked if self.requires_acess_to_other_agents is set

        :param other_agents_vector: List of all agents in the environment excluding `self`
        :param task: Task on which :param: self and :param: other_agents_vector
                     will act.
        :param num_envs: Number of parallel environments that will be spawned
                         by parallel task. Useful, for instance, to know how
                         many connections to generate on a NeuralNetServerHandler
        '''
        raise NotImplementedError('Should be implemented in subclass')

    def reset_after_episodes(self):
        '''
        TODO: improve wording
        Especially useful when running multiple episodes in parallel on a Task
        via `Task.run_episodes(...)`, as some episodes might be halted half-way through,
        corrupting an agent's internal state (i.e, with episode specific storages half-full,
        which need to be reseted when a new episode begins)
        '''
        pass

    @abstractmethod
    def clone(self):
        '''
        Function which should return an identical copy of the agent
        INCLUDING a deep copy of all the underlying objects used by the agent
        such as neural networks, storage, agent flags... etc
        '''
        raise NotImplementedError('To be implemented in Subclass')

    def start_server(self, num_connections: int):
        raise NotImplementedError('To be implemented in appropiate subclass')

    def close_server(self):
        self.server_handler.close_server()

    def __getstate__(self):
        '''
        Function invoked when pickling.

        Pickling processes handlers (a pointer to a multiprocessing.Process)
        is notoriously annoying in Python. Thus when pickling an agent
        the opinionated decision is taken to destroy the pointer to
        the `Agent.server_handler` to allow for easy pickling.
        Server can be started again via `Agent.start_server()`
        '''
        to_pickle_dict = self.__dict__
        if any(map(lambda key: key in to_pickle_dict, self.keys_to_not_pickle)):
            to_pickle_dict = self.__dict__.copy()
            for key in filter(lambda key: key in to_pickle_dict, self.keys_to_not_pickle):
                to_pickle_dict[key] = None
        return to_pickle_dict
