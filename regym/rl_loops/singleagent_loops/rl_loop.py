from typing import List, Tuple
from copy import deepcopy
import gym
import regym
from regym.rl_algorithms.agents import Agent


def run_episode(env: gym.Env, agent: Agent, training: bool, render_mode: str) -> Tuple:
    '''
    Runs a single episode of a single-agent rl loop until termination.
    :param env: OpenAI gym environment
    :param agent: Agent policy used to take actions in the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param render_mode: TODO: add rendering
    :returns: Episode trajectory. list of (o,a,r,o')
    '''
    observation = env.reset()
    done = False
    trajectory = []
    legal_actions: List = None
    while not done:
        action = agent.take_action(deepcopy(env)) if agent.requires_environment_model else agent.take_action(observation, legal_actions)
        succ_observation, reward, done, info = env.step(action)
        trajectory.append((observation, action, reward, succ_observation, done))
        if training: agent.handle_experience(observation, action, reward, succ_observation, done)
        observation = succ_observation

        if 'legal_actions' in info: legal_actions = info['legal_actions']

    return trajectory
