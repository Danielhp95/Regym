from unittest.mock import Mock, PropertyMock, MagicMock, patch

import numpy as np
import gym_connect4

from test_fixtures import Connect4Task

import regym
from regym.environments import EnvType
from regym.rl_algorithms import build_MCTS_Agent
from regym.rl_algorithms.agents import Agent, build_Deterministic_Agent, DeterministicAgent
from regym.rl_loops import Trajectory

from regym.rl_algorithms import build_Deterministic_Agent, build_MCTS_Agent


from regym.rl_loops.multiagent_loops.vectorenv_sequential_action_rl_loop import async_run_episode
from regym.rl_loops.multiagent_loops.sequential_action_rl_loop import propagate_experience, propagate_last_experience


def test_sequential_trajectories_feature_agent_predictions_single_env(Connect4Task):
    agent_1 = build_Deterministic_Agent(
        Connect4Task, {'action': 0}, 'Col-0-DeterministicAgent')
    agent_1.requires_opponents_prediction = True  # Required!
    agent_2 = build_Deterministic_Agent(
        Connect4Task, {'action': 1}, 'Col-0-DeterministicAgent')

    trajectory = Connect4Task.run_episode([agent_1, agent_2], training=False)

    expected_prediction_1 = {'a': 0, 'probs': [[1., 0., 0., 0., 0., 0., 0.]]}
    expected_prediction_2 = {'a': 1, 'probs': [[0., 1., 0., 0., 0., 0., 0.]]}
    expected_predictions = [expected_prediction_1,
                            expected_prediction_2]

    compare_trajectory_extra_info_against_expected(trajectory, expected_predictions)


def test_sequential_trajectories_feature_agent_predictions_multienv(Connect4Task):
    agent_1 = build_Deterministic_Agent(
        Connect4Task, {'action': 0}, 'Col-0-DeterministicAgent')
    agent_1.requires_opponents_prediction = True  # Required!
    agent_2 = build_Deterministic_Agent(
        Connect4Task, {'action': 1}, 'Col-0-DeterministicAgent')

    trajectories = Connect4Task.run_episodes([agent_1, agent_2], training=False,
                                             num_envs=2, num_episodes=2)

    # on single agents there's a batch dimension in 'probs', but not
    # on multiagent_loops. Does this matter?
    expected_prediction_1 = {'a': 0, 'probs': [1., 0., 0., 0., 0., 0., 0.]}
    expected_prediction_2 = {'a': 1, 'probs': [0., 1., 0., 0., 0., 0., 0.]}
    expected_predictions = [expected_prediction_1, expected_prediction_2]

    for trajectory in trajectories:
        compare_trajectory_extra_info_against_expected(trajectory, expected_predictions)


def test_agents_in_sequential_environments_handle_experiences_with_extra_info_single_env(Connect4Task):
    '''
    In this test we want to ensure that when agents process experiences
    via `Agent.handle_experience(...)` calls, they obtain the are able
    to observe the `predicions` of other agents.

    There are 2 cases to consider:
        - Handling an experience in the middle of a trajectory
        - Handling an experience when the episode just finshed and some agents
          need to process the last (terminal) timestep
    '''
    mock_agent_1 = Mock(spec=DeterministicAgent)
    mock_agent_2 = Mock(spec=DeterministicAgent)
    agent_vector = [mock_agent_1, mock_agent_2]

    mock_agent_1.requires_opponents_prediction = True
    mock_agent_1.training = True
    mock_agent_2.requires_opponents_prediction = False
    mock_agent_2.training = True

    prediction_1 = {'a': 0, 'probs': [1., 0., 0., 0., 0., 0., 0.]}
    prediction_2 = {'a': 1, 'probs': [0., 1., 0., 0., 0., 0., 0.]}
    predictions = [prediction_1, prediction_2]


    '''
    Creates a trajectory for the game of Connect4 looks like this
        Total timesteps 7. P1 (x) actions: 4. P2 (o) actions: 3.

    Board:     |              |
               |              |
               |x             |
               |x o           |
               |x o           |
               |x o . . . . . |
               |--------------|
    '''
    sample_trajectory = Trajectory(
        env_type=EnvType.MULTIAGENT_SEQUENTIAL_ACTION, num_agents=2)

    o, a, r, succ_o, done = [None, None], None, [0, 0], [None, None], False

    sample_trajectory.add_timestep(
        o, a, r, succ_o, done, acting_agents=[0],
        extra_info={0: predictions[0]})
    sample_trajectory.add_timestep(
        o, a, r, succ_o, done, acting_agents=[1],
        extra_info={1: predictions[1]})

    # Update agent 0 
    propagate_experience(agent_vector, sample_trajectory)

    sample_trajectory.add_timestep(
        o, a, r, succ_o, done, acting_agents=[0],
        extra_info={0: predictions[0]})

    # Update agent 1
    propagate_experience(agent_vector, sample_trajectory)

    sample_trajectory.add_timestep(
        o, a, r, succ_o, done, acting_agents=[1],
        extra_info={1: predictions[1]})

    # Update agent 0
    propagate_experience(agent_vector, sample_trajectory)

    sample_trajectory.add_timestep(
        o, a, r, succ_o, done, acting_agents=[0],
        extra_info={0: predictions[0]})

    # Update agent 1
    propagate_experience(agent_vector, sample_trajectory)

    sample_trajectory.add_timestep(
        o, a, r, succ_o, done, acting_agents=[1],
        extra_info={1: predictions[1]})

    # Update agent 0
    propagate_experience(agent_vector, sample_trajectory)

    sample_trajectory.add_timestep(
        o, a, [1, -1], succ_o, done, acting_agents=[0],
        extra_info={0: predictions[0]})

    # Update player 1
    propagate_experience(agent_vector, sample_trajectory)

    # Episode termination
    # Update player 0 (After done flag)
    propagate_last_experience(agent_vector, sample_trajectory)


def compare_trajectory_extra_info_against_expected(trajectory, expected_predictions):
    for timestep in trajectory:
        # Only one agent acts at a time in Connect4
        a_i = timestep.acting_agents[0]
        actual_prediction = timestep.extra_info[a_i]

        assert 'a' in actual_prediction
        assert 'probs' in actual_prediction
        assert actual_prediction['a'] == expected_predictions[a_i]['a']
        np.testing.assert_array_equal(
            actual_prediction['probs'], expected_predictions[a_i]['probs'])

