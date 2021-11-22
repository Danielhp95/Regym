import pytest

from regym.environments import generate_task
from regym.environments import EnvType


@pytest.fixture
def Connect4Task():
    import gym_connect4
    return generate_task('Connect4-v0', EnvType.MULTIAGENT_SEQUENTIAL_ACTION)
