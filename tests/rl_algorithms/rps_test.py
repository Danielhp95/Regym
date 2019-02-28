from tqdm import tqdm
import gym
from multiagent_loops import simultaneous_action_rl_loop


def learns_against_fixed_opponent_RPS(agent, fixed_opponent, training_episodes, inference_percentage, reward_threshold):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    env = gym.make('RockPaperScissors-v0')
    maximum_average_reward, reward_threshold = 1.0, 0.10
    inference_percentage = 0.9

    agent_vector = [agent, fixed_opponent]
    trajectories = list()
    pbar = tqdm(range(training_episodes))
    for e in pbar:
        trajectory = simultaneous_action_rl_loop.run_episode(env, agent_vector, training=True)
        trajectories.append(trajectory)
        avg_trajectory_reward = sum(map(lambda experience: experience[2][0], trajectory)) / len(trajectory)
        pbar.set_description(f'Training {agent.name} againts {fixed_opponent.name}. Last avg reward: {avg_trajectory_reward}')

    average_rewards = [sum(map(lambda experience: experience[2][0], t)) / len(t) for t in trajectories]
    inference_rewards = average_rewards[int(training_episodes * inference_percentage):]
    average_inference_reward = sum(inference_rewards) / len(inference_rewards)
    assert average_inference_reward  >= maximum_average_reward - reward_threshold
