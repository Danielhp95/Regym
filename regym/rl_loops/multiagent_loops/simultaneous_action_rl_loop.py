from typing import List, Tuple
from copy import deepcopy

from PIL import Image
import gym

import regym
from regym.rl_loops import Trajectory


def run_episode(env: gym.Env, agent_vector: List['Agent'],
                training: bool, render_mode: str = '',
                save_gif=False) -> Trajectory:
    '''
    Runs a single multi-agent rl loop until termination where each agent
    takes an action simulatenously.
    The observations vector is of length n, where n is the number of agents
    observations[i] corresponds to the oberservation of agent i.
    :param env: OpenAI gym environment
    :param agent_vector: Vector containing the agent for each agent in the environment
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :returns: Episode trajectory (o,a,r,o',d)
    '''
    observations = env.reset()
    done = False
    trajectory = Trajectory(env_type=regym.environments.EnvType.MULTIAGENT_SIMULTANEOUS_ACTION,
                            num_agents=len(agent_vector))
    iteration = 0
    acting_agents = list(range(len(agent_vector)))  # ASSUMPTION All agents act at all timesteps
    # Unfortunately, OpenAIGym does not have a standardized interface
    # To support which actions are legal at an initial state. These can only be extracted
    # via the "info" dictionary given by the env.step(...) function
    # Thus: Assumption: all actions are permitted on the first state
    legal_actions: List = None
    while not done:
        if render_mode: render(env, render_mode)

        iteration += 1
        action_vector = [
                agent.model_based_take_action(deepcopy(env), observations[i], player_index=i) if agent.requires_environment_model \
                        else agent.model_free_take_action(observations[i], legal_actions)
                for i, agent in enumerate(agent_vector)]
        succ_observations, reward_vector, done, info = env.step(action_vector)
        trajectory.add_timestep(observations, action_vector, reward_vector,
                                succ_observations, done, acting_agents)
        if training:
            for i, agent in enumerate(agent_vector):
                agent.handle_experience(observations[i], action_vector[i], reward_vector[i], succ_observations[i], done)
        observations = succ_observations

        if 'legal_actions' in info: legal_actions = info['legal_actions']

    return trajectory


def render(env: gym.Env, render_mode: str):
    if render_mode == 'string': print(env.render(render_mode))
    elif render_mode == 'rgb': env.render('rgb')
    else: NotImplementedError(f'Rendering for render_mode {render_mode} not currently supported')
