from regym.rl_algorithms import build_Random_Agent

from regym.tests.test_utils.fixed_length_dummy_env import FixedLengthDummyTask


def test_correct_number_of_experiences_are_handled(FixedLengthDummyTask):
    rando_1 = build_Random_Agent(FixedLengthDummyTask, {}, 'Rando-0')
    rando_2 = build_Random_Agent(FixedLengthDummyTask, {}, 'Rando-1')

    num_episodes = 1

    trajectory = FixedLengthDummyTask.run_episodes(
        agent_vector=[rando_1, rando_2],
        num_episodes=num_episodes,
        num_envs=1,
        training=True
    )

    episode_length = FixedLengthDummyTask.env.episode_length

    assert rando_1.finished_episodes == num_episodes
    assert rando_1.handled_experiences == 2
    assert rando_2.handled_experiences == 1
    assert rando_2.finished_episodes == num_episodes
    assert FixedLengthDummyTask.total_episodes_run == num_episodes
