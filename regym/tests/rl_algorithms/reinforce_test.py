from test_fixtures import reinforce_config_dict, CartPoleTask, RPSTask

from regym.rl_algorithms.agents import build_Reinforce_Agent
from regym.rl_algorithms import rockAgent
from regym.rl_loops.singleagent_loops.rl_loop import run_episode
from regym.rl_loops.multiagent_loops import simultaneous_action_rl_loop


def test_reinforce_can_take_actions(CartPoleTask, reinforce_config_dict):
    env = CartPoleTask.env
    agent = build_Reinforce_Agent(CartPoleTask, reinforce_config_dict, 'Test-Reinforce')
    number_of_actions = 5
    for i in range(number_of_actions):
        random_observation = env.observation_space.sample()
        action = agent.take_action(random_observation)
        assert env.action_space.contains(action)


def test_learns_to_solve_cartpole(CartPoleTask, reinforce_config_dict):
    agent = build_Reinforce_Agent(CartPoleTask, reinforce_config_dict, 'Test-Reinforce')
    assert agent.training, 'Agent should be training in order to solve test environment'
    import tqdm
    progress_bar = tqdm.tqdm(range(2000))
    for _ in progress_bar:
        trajectory = run_episode(CartPoleTask.env, agent, training=True)
        avg_trajectory_reward = sum(map(lambda experience: experience[2], trajectory)) / len(trajectory)
        progress_bar.set_description(f'{agent.name} in {CartPoleTask.env.spec.id}. Episode length: {len(trajectory)}')
