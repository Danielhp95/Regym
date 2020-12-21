from functools import partial

import numpy as np
import gym
import gym_rock_paper_scissors
import gym_connect4

from regym.environments import generate_task, EnvType
from regym.environments.wrappers import FrameStack

from regym.environments.tasks import RegymAsyncVectorEnv


def test_can_stack_frames_singleagent_env():
    num_stack = 3
    frame_stack = partial(FrameStack, num_stack=num_stack)

    pendulum_task = generate_task('Pendulum-v0')
    stack_pendulum_task = generate_task('Pendulum-v0',
                                        wrappers=[frame_stack])

    assert stack_pendulum_task.observation_dim == (num_stack, *pendulum_task.observation_dim)


def test_can_stack_frames_sequential_multiagent_env():
    num_stack = 4
    frame_stack = partial(FrameStack, num_stack=num_stack)

    connect_4_task = generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)
    stack_connect_4_task = generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION,
                                         wrappers=[frame_stack])
    assert stack_connect_4_task.observation_dim == (num_stack, *connect_4_task.observation_dim)

    num_envs = 3
    vector_env = RegymAsyncVectorEnv(
        stack_connect_4_task.name,
        num_envs=num_envs,
        wrappers=[frame_stack]
    )

    actual_obs = vector_env.reset()

    # Standard Connect4 dimensions is (3, 7, 6)
    # NOTE: Think of board as being sideways (chips fall right-to-left)
    single_env_initial_observation = np.array(
         [[[1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.]],

          [[0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.]],

          [[0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0.]]]
    )

    # We extend by number of stacked frames
    # So that per environment observation shape is (num_stacks, 3, 7, 6)
    stacked_single_env_initial_observation = np.array(
        [single_env_initial_observation for _ in range(num_stack)]
    )

    # We extend by number of environments
    # So that each agent receives observation of shape (num_envs, num_stack, 3, 7, 6)
    expected_player_obs = np.array(
        [stacked_single_env_initial_observation for _ in range(num_envs)]
    )

    num_agents = 2
    for i in range(num_agents):
        np.testing.assert_array_equal(expected_player_obs, actual_obs[i])
