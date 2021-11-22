from typing import List, Tuple, Any, Optional
from copy import deepcopy

import tqdm
import gym

import regym
from regym.rl_algorithms.agents import Agent
from regym.environments import EnvType
from regym.environments.tasks import RegymAsyncVectorEnv
from regym.rl_loops.utils import update_trajectories, update_finished_trajectories
from regym.rl_loops.trajectory import Trajectory


def run_episode(env: gym.Env, agent: Agent, training: bool, render_mode: str) \
                -> List[Trajectory]:
    '''
    Runs a single episode of a single-agent rl loop until termination.
    :param env: OpenAI gym environment
    :param agent: Agent policy used to take actions in the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param render_mode: Feature not implemented (yet!)
    :returns: Episode trajectory. list of (o,a,r,o')
    '''
    observation = env.reset()
    done = False
    trajectory = Trajectory(env_type=EnvType.SINGLE_AGENT)
    legal_actions: List = None
    while not done:
        if agent.requires_environment_model:
            action = agent.model_based_take_action(deepcopy(env), observation, player_index=0)
        else:
            action = agent.model_free_take_action(observation, legal_actions)
        succ_observation, reward, done, info = env.step(action)
        trajectory.add_timestep(observation, action, reward, succ_observation, done)
        if training: agent.handle_experience(observation, action, reward, succ_observation, done)
        observation = succ_observation

        if 'legal_actions' in info: legal_actions = info['legal_actions']

    return trajectory


def async_run_episode(env: RegymAsyncVectorEnv, agent: Agent, training: bool,
                      num_episodes: int, show_progress=True) \
                      -> List[Trajectory]:
    '''
    TODO

    NOTE: Unlike regular gym.Env. RegymAsyncVectorEnv resets an underlying
    environment once its ongoing episode finishes.
    :param env: OpenAI gym environment
    :param agent: Agent policy used to take actions in the environment
                  and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :returns: List of episode trajectories: list of (list of (o,a,r,o'))
    '''
    ongoing_trajectories = [Trajectory(env_type=EnvType.SINGLE_AGENT)
                            for _ in range(env.num_envs)]
    finished_trajectories = []

    obs = env.reset()
    legal_actions: List[List] = None  # Revise

    if show_progress: progress_bar = tqdm.tqdm(total=num_episodes)

    while len(finished_trajectories) < num_episodes:
        action_vector = choose_action(agent, env, obs, legal_actions)
        succ_obs, rewards, dones, infos = env.step(action_vector)

        update_trajectories(ongoing_trajectories, action_vector, obs,
                            rewards, succ_obs, dones)

        if training: update_agent(agent, ongoing_trajectories)

        obs = succ_obs
        if 'legal_actions' in infos[0]:
            legal_actions = [info['legal_actions'] for info in infos]

        (ongoing_trajectories, finished_trajectories) = handle_finished_episodes(
                 ongoing_trajectories, finished_trajectories, dones,
                 progress_bar if show_progress else None)
    if show_progress: progress_bar.close()
    return finished_trajectories


def choose_action(agent: 'Agent', env: RegymAsyncVectorEnv, observation,
                  legal_actions: List[List[int]]) -> List:
    ''' Takes an action from the agent, conditioned on :param: observation
        and legal_actions '''
    if agent.requires_environment_model:
        raise NotImplementedError('Model-based take_action not implemented for singleagent async run episode')
    else:
        action_vector = agent.model_free_take_action(
                observation, legal_actions, multi_action=True)
    return action_vector


def update_agent(agent: 'Agent', ongoing_trajectories: List[Trajectory]):
    ''' Propagates latest experiences from :param: ongoing_trajectories
    '''
    experiences = [(t[-1].observation, t[-1].action, t[-1].reward,
                    t[-1].succ_observation, t[-1].done, t[-1].extra_info)
                   for t in ongoing_trajectories]
    agent.handle_multiple_experiences(experiences,
                                      list(range(len(ongoing_trajectories))))


def handle_finished_episodes(ongoing_trajectories: List[Trajectory],
                             finished_trajectories: List[Trajectory],
                             dones: List[bool],
                             progress_bar: Optional[tqdm.std.tqdm]) \
                             -> Tuple[List, List]:
    ''' Copies finished :param: ongoing_trajectories into
        :param: finished_trajectories as dictated by :param: done
        also updating :param: progress_bar if required '''
    done_envs = [i for i in range(len(dones)) if dones[i]]
    if len(done_envs) > 0:
        progress_bar.update(len(done_envs))
        ongoing_trajectories, finished_trajectories = update_finished_trajectories(
                ongoing_trajectories, finished_trajectories, done_envs)
    return ongoing_trajectories, finished_trajectories
