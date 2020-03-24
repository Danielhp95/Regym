from typing import List, Tuple
from copy import deepcopy

from PIL import Image

import gym
import regym
from regym.rl_algorithms.agents import Agent


def run_episode(env: gym.Env, agent_vector: List[Agent], training: bool, render_mode: str = '', save_gif=True) -> Tuple:
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
    trajectory = []
    iteration = 0
    images = []

    while not done:
        if render_mode != '': rendered_state = env.render(render_mode)
        if render_mode == 'string': print(rendered_state)
        elif render_mode == 'rgb': env.render('rgb')
        if save_gif:
            pass #  images.append(Image.fromarray(env.render('rgb')))

        iteration += 1
        action_vector = [
                agent.take_action(deepcopy(env), i) if agent.requires_environment_model else agent.take_action(observations[i])
                for i, agent in enumerate(agent_vector)]
        succ_observations, reward_vector, done, info = env.step(action_vector)
        trajectory.append((observations, action_vector, reward_vector, succ_observations, done))
        if training:
            for i, agent in enumerate(agent_vector): agent.handle_experience(observations[i], action_vector[i], reward_vector[i], succ_observations[i], done)
        observations = succ_observations

    if len(images) > 0:
        gif_name = env.spec.id + '_' + '_vs_'.join([a.name for a in agent_vector])
        generate_gif_from_images(gif_name, images)

    return trajectory


def generate_gif_from_images(gif_name: str, images: List[Image.Image]):
    images[0].save(f'{gif_name}.gif',
               save_all=True,
               append_images=images[1:],
               duration=100,
               loop=0)
