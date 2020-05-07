from typing import List, Tuple, Any
from copy import deepcopy
import gym
import regym
from regym.rl_algorithms.agents import Agent

from regym.environments.tasks import RegymAsyncVectorEnv


def run_episode(env: gym.Env, agent: Agent, training: bool, render_mode: str) \
                -> List[Tuple[Any, Any, Any, Any, bool]]:
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
        if agent.requires_environment_model:
            action = agent.model_based_take_action(deepcopy(env), observation, player_index=0)
        else:
            action = agent.model_free_take_action(observation, legal_actions)
        succ_observation, reward, done, info = env.step(action)
        trajectory.append((observation, action, reward, succ_observation, done))
        if training: agent.handle_experience(observation, action, reward, succ_observation, done)
        observation = succ_observation

        if 'legal_actions' in info: legal_actions = info['legal_actions']

    return trajectory


def async_run_episode(env: RegymAsyncVectorEnv, agent: Agent, training: bool,
                      num_episodes: int) \
                      -> List[List[Tuple[Any, Any, Any, Any, bool]]]:
    '''
    TODO

    NOTE: Unline regular gym.Env. RegymAsyncVectorEnv resets an underlying
    environment once its ongoing episode finishes.
    '''
    ongoing_trajectories: List[List[Tuple[Any, Any, Any, Any, bool]]]
    ongoing_trajectories = [[] for _ in range(env.num_envs)]
    finished_trajectories = []

    done_envs = lambda dones: [i for i in range(len(dones)) if dones[i]]

    obs = env.reset()
    legal_actions: List[List] = None  # Revise
    while len(finished_trajectories) < num_episodes:
        action_vector = None
        if agent.requires_environment_model:
            raise NotImplementedError('Gimme a minute')
        else:
            action_vector = agent.model_free_take_action(obs, legal_actions)
        succ_obs, rewards, dones, infos = env.step(action_vector)

        if 'legal_actions' in infos[0]:
            legal_actions = [info['legal_actions'] for info in infos]

        for i in range(env.num_envs):
            e = (obs[i], action_vector[i], rewards[i], succ_obs[i], dones[i])
            ongoing_trajectories[i] += [e]

        obs = succ_obs

        for env_i in done_envs(dones):
            finished_trajectories += [ongoing_trajectories[env_i]]
            ongoing_trajectories[env_i] = []
    return finished_trajectories
