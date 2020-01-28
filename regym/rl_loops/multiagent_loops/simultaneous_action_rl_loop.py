from typing import List, Tuple
from copy import deepcopy
import gym
import regym
from regym.rl_algorithms.agents import Agent

def run_episode(env: gym.Env, agent_vector: List[Agent], training: bool) -> Tuple:
    '''
    Runs a single multi-agent rl loop until termination where each agent
    takes an action simulatenously.
    The observations vector is of length n, where n is the number of agents
    observations[i] corresponds to the oberservation of agent i.
    :param env: OpenAI gym environment
    :param agent_vector: Vector containing the agent for each agent in the environment
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :returns: Episode trajectory (o,a,r,o')
    '''
    observations = env.reset()
    done = False
    trajectory = []
    while not done:
        action_vector = [
                agent.take_action(deepcopy(env)) if agent.requires_environment_model else agent.take_action(observations[i])
                for i, agent in enumerate(agent_vector)]
        succ_observations, reward_vector, done, info = env.step(action_vector)
        trajectory.append((observations, action_vector, reward_vector, succ_observations, done))
        if training:
            for i, agent in enumerate(agent_vector): agent.handle_experience(observations[i], action_vector[i], reward_vector[i], succ_observations[i], done)
        observations = succ_observations

    return trajectory
