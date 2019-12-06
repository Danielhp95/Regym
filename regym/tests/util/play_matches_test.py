import pytest
import numpy as np

from regym.rl_algorithms import rockAgent, scissorsAgent 
from regym.environments import generate_task
from regym.environments import EnvType
from regym.util.play_matches import play_multiple_matches


@pytest.fixture
def RPS_task():
    import gym_rock_paper_scissors
    return generate_task('RockPaperScissors-v0', EnvType.MULTIAGENT_SIMULTANEOUS_ACTION)


@pytest.fixture
def Kuhn_task():
    import gym_kuhn_poker
    return generate_task('KuhnPoker-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)


def test_can_play_simultaneous_action_environments(RPS_task):
    agent_vector = [rockAgent, scissorsAgent]
    play_matches_given_task_and_agent_vector(RPS_task, agent_vector)


def test_can_play_sequential_action_environments(Kuhn_task):
    class FixedAgent():
        def __init__(self, action):
            self.name = f'FixedAction: {action}'
            self.action = action

        def take_action(self, *args):
            return self.action

        def handle_experience(self, *args):
            pass
    agent_vector = [FixedAgent(1), FixedAgent(0)]
    play_matches_given_task_and_agent_vector(Kuhn_task, agent_vector)

def play_matches_given_task_and_agent_vector(task, agent_vector):
    expected_winrates = [1., 0.]
    number_matches = 10

    winrates, trajectories = play_multiple_matches(task, agent_vector,
                                                   n_matches=number_matches,
                                                   keep_trajectories=True)

    assert len(trajectories) == number_matches
    assert task.total_episodes_run == number_matches
    np.testing.assert_array_equal(expected_winrates, winrates)
