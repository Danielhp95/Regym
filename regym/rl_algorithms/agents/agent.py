from abc import ABC, abstractmethod

class Agent(ABC):
    '''
    This Agent class is an abstract interface from to subclass all other agents.

    An agent is an entity that communicates with an environment for two purposes:
        - To take actions in said environment using a policy.
        - To collect experiences with which to feed to an underlying algorithm that will
          update the Agent's policy. An experience is a Tuple of:
            - (state, action, reward, successor state, done)

    Each algorithm (PPO, DQN, REINFORCE) can heavily differ in:
        - How the environment experiences are handled
          (i.e. when they are removed from memory, which experiences to keep)
        - How actions are taken (i.e using a tabular method, using neural networks)
          Or even using an environment model.

    Finally, agents can be model-based or model-free. By default agents are model-free,
    as specified by their property `Agent.requires_environment_model = False`.
    The difference between model-based and model-free agents is that the former
    receive a copy of the environment every time they take an action so that they
    can perform search on them. Model-free agents instead receive only a copy of the
    environment state.
    '''

    def __init__(self, name: str, requires_environment_model=False):
        '''
        By default agents do not require an environment model.
        What this means is that agents take actions based on the state of
        the environment (although agents may internally store more information).
        Agents which do depend on an environment model will receive a copy of the
        environment every time they are asked to return an action.

        :param name: String identifier, Agents are named, for all great creations should be named
        '''
        self.requires_environment_model = requires_environment_model
        self.name = name
        self.training = True
        self.handled_experiences = 0

    @abstractmethod
    def take_action(self, state_or_environment):
        '''
        This function is called inside of an regym.rl_loops, asking
        the Agent to take an action at a given state in the environment
        so that the environment model may move forward.

        :param state_or_environment: Either the current state of the environment
         or a copy of the environment being played. Whether an agent receives a
         state or an environment depends on its flag `Agent.requires_environment_model`.
        '''
        pass

    @abstractmethod
    def handle_experience(self, s, a, r, succ_s, done=False):
        '''
        Processes a single 'experience' (defined by the parameters of this function),
        which is the main method of gathering data of an RL algorithm.
        NOTE: Unless this agent's 'training' flag is set to True, this function will not do anything.

        :param s:      Environment state
        :param a:      action taken by this agent at :param s: state
        :param r:      reward obtained by this agent after taking :param a: action at :param s: state
        :param succ_s: Environment state reached after after taking :param a: action at :param s: state
        :param done:   Boolean representing whether the environment episode has finished
        '''
        self.handled_experiences += 1

    @abstractmethod
    def clone(self):
        '''
        Function which should return an identical copy of the agent
        INCLUDING a deep copy of all the underlying objects used by the agent
        such as neural networks, storage, agent flags... etc
        '''
        pass
