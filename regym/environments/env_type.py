from enum import Enum


class EnvType(Enum):
    '''
    Enumerator representing what kind of environment a task will deal with.
    Useful because different environments (simulatenous vs sequential) require
    a different underlying mathematical construct to simulate an episode
    '''
    SINGLE_AGENT = 'single-agent'
    MULTIAGENT_SIMULTANEOUS_ACTION = 'multiagent-simultaneous'
    MULTIAGENT_SEQUENTIAL_ACTION = 'multiagent-sequential'
