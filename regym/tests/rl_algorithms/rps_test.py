from tqdm import tqdm
import gym
from regym.rl_loops.multiagent_loops import simultaneous_action_rl_loop
from environments import ParallelEnv


def learns_against_fixed_opponent_RPS(agent, fixed_opponent, total_episodes, training_percentage, reward_threshold):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays rock in rock paper scissors.
    i.e from random, learns to play only (or mostly) paper
    '''
    env = gym.make('RockPaperScissors-v0')
    maximum_average_reward = 1.0

    training_episodes = int(total_episodes * training_percentage)
    inference_episodes = total_episodes - training_episodes

    training_trajectories = simulate(env, agent, fixed_opponent, episodes=training_episodes, training=True)
    agent.training = False
    inference_trajectories = simulate(env, agent, fixed_opponent, episodes=inference_episodes, training=False)

    average_inference_rewards = [sum(map(lambda experience: experience[2][0], t)) / len(t) for t in inference_trajectories]
    average_inference_reward = sum(average_inference_rewards) / len(average_inference_rewards)
    assert average_inference_reward  >= maximum_average_reward - reward_threshold


def simulate(env, agent, fixed_opponent, episodes, training):
    agent_vector = [agent, fixed_opponent]
    trajectories = list()
    mode = 'Training' if training else 'Inference'
    progress_bar = tqdm(range(episodes))
    for e in progress_bar:
        trajectory = simultaneous_action_rl_loop.run_episode(env, agent_vector, training=training)
        trajectories.append(trajectory)
        avg_trajectory_reward = sum(map(lambda experience: experience[2][0], trajectory)) / len(trajectory)
        progress_bar.set_description(f'{mode} {agent.name} against {fixed_opponent.name}. Last avg reward: {avg_trajectory_reward}')
    return trajectories


def learns_against_fixed_opponent_RPS_parallel(agent, fixed_opponent, total_episodes, training_percentage, reward_threshold_percentage, envname='RockPaperScissors-v0', nbr_parallel_env=2):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays randomly.
    i.e from random, learns to play only (or mostly) paper
    '''
    env = ParallelEnv(envname, nbr_parallel_env)
    maximum_average_reward = 10.0

    training_episodes = int(total_episodes * training_percentage)
    inference_episodes = total_episodes - training_episodes

    training_trajectories = simulate_parallel(env, agent, fixed_opponent, episodes=training_episodes, training=True)

    agent.training = False

    env = gym.make(envname)
    inference_trajectories = simulate(env, agent, fixed_opponent, episodes=inference_episodes, training=False)

    average_inference_rewards = [sum(map(lambda experience: experience[2][0], t)) for t in inference_trajectories]
    average_inference_reward = sum(average_inference_rewards) / len(average_inference_rewards)
    assert average_inference_reward  >= maximum_average_reward*reward_threshold_percentage

def simulate_parallel(env, agent, fixed_opponent, episodes, training):
    agent_vector = [agent, fixed_opponent]
    trajectories = list()
    mode = 'Training' if training else 'Inference'
    progress_bar = tqdm(range(episodes))
    for e in progress_bar:
        per_actor_trajectories = simultaneous_action_rl_loop.run_episode_parallel(env, agent_vector, training=training, self_play=False)
        trajectory = []
        for t in per_actor_trajectories.values():
            trajectories.append(t)
            for exp in t:
                trajectory.append( exp)
        avg_trajectory_reward = sum(map(lambda experience: experience[2][0], trajectory)) / len(trajectory)
        progress_bar.set_description(f'{mode} {agent.name} against {fixed_opponent.name}. Last avg reward: {avg_trajectory_reward}')
    return trajectories
