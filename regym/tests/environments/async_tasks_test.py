import time
import multiprocessing

import pytest
import gym_connect4

import regym
from regym.environments import generate_task, EnvType
from regym.rl_algorithms import build_Random_Agent


@pytest.mark.parametrize('env_name', ['CartPole-v0'])
def test_can_run_multiple_async_episodes_of_singleagent_task(env_name):
    run_parallel_task_with_random_agent(env_name, EnvType.SINGLE_AGENT,
                                        num_envs=4, num_episodes=2)


@pytest.mark.parametrize('env_name', ['CartPole-v0'])
def test_singleagent_tasks_run_faster_on_parallel(env_name):
    task = generate_task(env_name, EnvType.SINGLE_AGENT)
    random_agent = build_Random_Agent(task, {}, 'Test-Random')

    start = time.time()
    num_episodes = 200
    num_envs = 1
    trajectories = task.run_episodes([random_agent], num_episodes=num_episodes,
                                     num_envs=num_envs, training=False)
    total_single = time.time() - start

    start = time.time()
    num_envs = multiprocessing.cpu_count()
    trajectories = task.run_episodes([random_agent], num_episodes=num_episodes,
                                     num_envs=num_envs, training=False)
    total_multiple = time.time() - start

    assert total_multiple < total_single


@pytest.mark.parametrize('env_name', ['Connect4-v0'])
def test_can_run_multiple_async_episodes_of_multiagent_simultaneous_task(env_name):
    assert False


@pytest.mark.parametrize('env_name', ['Connect4-v0'])
def test_can_run_multiple_async_episodes_of_multiagent_sequential_task(env_name):
    run_parallel_task_with_random_agent(env_name, EnvType.MULTIAGENT_SEQUENTIAL_ACTION,
                                        num_envs=4, num_episodes=2)


@pytest.mark.parametrize('env_name', ['Connect4-v0'])
def test_multiagent_sequential_tasks_run_faster_on_parallel(env_name):
    task = generate_task(env_name, EnvType.MULTIAGENT_SEQUENTIAL_ACTION)
    random_agent = build_Random_Agent(task, {}, 'Test-Random')

    start = time.time()
    num_episodes = 200
    num_envs = 1
    _ = task.run_episodes([random_agent, random_agent], num_episodes=num_episodes,
                          num_envs=num_envs, training=False)
    total_single = time.time() - start

    start = time.time()
    num_envs = multiprocessing.cpu_count()
    _ = task.run_episodes([random_agent, random_agent], num_episodes=num_episodes,
                          num_envs=num_envs, training=False)
    total_multiple = time.time() - start
    print('Parallel: ', total_multiple, 'Sequential: ', total_single,
          'Diff: ', total_single - total_multiple)
    assert total_multiple < total_single


@pytest.mark.parametrize('env_name', ['CartPole-v0'])
def test_multiagent_simultaneous_tasks_run_faster_on_parallel(env_name):
    assert False


def run_parallel_task_with_random_agent(env_name, env_type,
                                        num_envs, num_episodes):
    task = generate_task(env_name, env_type)
    agent_vector = [build_Random_Agent(task, {}, 'Test-Random')
                    for _ in range(task.num_agents)]

    # The number of environments is larger than number of
    # episodes because we want to test if we can generate
    # a specific number of trajectories regardless of the
    # Number of environments used to generate them
    trajectories = task.run_episodes(agent_vector, num_episodes=num_episodes,
                                     num_envs=num_envs, training=False)

    # We have the exact number of trajectories we asked for
    assert len(trajectories) == num_episodes
    # All trajectories finish with a "done" flag
    assert all([t[-1][-1] for t in trajectories])
    if env_type == EnvType.SINGLE_AGENT:
        # Observation and succ_observation are the same
        # ASSUMPTION: observation and succ_observation are numpy array
        assert all([(ex_1[-2] == ex_2[0]).all()
                    for t in trajectories
                    for ex_1, ex_2 in zip(t, t[1:])])
    else:
        # TODO: think of something
        pass

# Add tests for MCTS agents (agents that require environment)
