import os
import sys
sys.path.append(os.path.abspath('..'))
from training_schemes import EmptySelfPlay, NaiveSelfPlay, HalfHistorySelfPlay, FullHistorySelfPlay

from rl_algorithms import build_DQN_Agent
from rl_algorithms import build_TabularQ_Agent
#from rl_algorithms import build_PPO_Agent
from rl_algorithms import rockAgent, paperAgent, scissorsAgent
from rl_algorithms import AgentHook

from . import gym_utils


def initialize_training_schemes(training_schemes_cli):
    def parse_training_scheme(training_scheme):
        if training_scheme.lower() == 'fullhistoryselfplay': return FullHistorySelfPlay
        elif training_scheme.lower() == 'halfhistoryselfplay': return HalfHistorySelfPlay
        elif training_scheme.lower() == 'naiveselfplay': return NaiveSelfPlay
        else: raise ValueError('Unknown training scheme {}. Try defining it inside this script.'.format(training_scheme))
    return [parse_training_scheme(t_s) for t_s in training_schemes_cli]


def initialize_algorithms(environment, algorithms_cli, base_path):
    def parse_algorithm(algorithm, env):
        if algorithm.lower() == 'tabularqlearning':
            return build_TabularQ_Agent(env.state_space_size, env.action_space_size, env.hash_state)
        if algorithm.lower() == 'deepqlearning':
            # TODO Should use_cuda be pased as parameter?
            return build_DQN_Agent(state_space_size=env.state_space_size, action_space_size=env.action_space_size, hash_function=env.hash_state, double=False, dueling=False, use_cuda=False)
        #if algorithm.lower() == 'ppo':
        #    return build_PPO_Agent(env)
        else: raise ValueError('Unknown algorithm {}. Try defining it inside this script.'.format(algorithm))

    return [parse_algorithm(algorithm, environment) for algorithm in algorithms_cli], [os.path.join(base_path, algorithm.lower())+'.pt' for algorithm in algorithms_cli]


def initialize_fixed_agents(fixed_agents_cli):
    def parse_fixed_agent(agent):
        if agent.lower() == 'rockagent': return AgentHook(rockAgent)
        elif agent.lower() == 'paperagent': return AgentHook(paperAgent)
        elif agent.lower() == 'scissorsagent': return AgentHook(scissorsAgent)
        else: raise ValueError('Unknown fixed agent {}. Try defining it inside this script.'.format(agent))
    return [parse_fixed_agent(agent) for agent in fixed_agents_cli]
