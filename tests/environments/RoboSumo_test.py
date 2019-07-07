from tqdm import tqdm
import gym
from regym.environments import ParallelEnv, EnvironmentCreator
from regym.rl_loops.multiagent_loops import simultaneous_action_rl_loop
from regym.rl_algorithms import AgentHook


def learns_against_fixed_opponent_RoboSumo(agent, fixed_opponent, total_episodes, training_percentage, reward_threshold_percentage, envname='RoboschoolSumo-v0', save=False):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays randomly.
    i.e from random, learns to play only (or mostly) paper
    '''
    env = gym.make(envname)
    maximum_average_reward = 20.0

    training_episodes = int(total_episodes * training_percentage)
    inference_episodes = total_episodes - training_episodes

    training_trajectories = simulate(env, agent, fixed_opponent, episodes=training_episodes, training=True)
    
    agent.training = False
    if save:
        agent_hook = AgentHook(agent.clone(), save_path='/tmp/test_ddpg_{}.agent'.format(envname))

    inference_trajectories = simulate(env, agent, fixed_opponent, episodes=inference_episodes, training=False)

    average_inference_rewards = [sum(map(lambda experience: experience[2][0], t)) for t in inference_trajectories]
    average_inference_reward = sum(average_inference_rewards) / len(average_inference_rewards)
    assert average_inference_reward  >= maximum_average_reward*reward_threshold_percentage

def record_against_fixed_opponent_RoboSumo(agent, fixed_opponent, envname='RoboschoolSumo-v0'):
    env = gym.make(envname)
    inference_trajectories = simulate(env, agent, fixed_opponent, episodes=1, training=False, record=envname)

def simulate(env, agent, fixed_opponent, episodes, training, record=None):
    agent_vector = [agent, fixed_opponent]
    trajectories = list()
    mode = 'Training' if training else 'Inference'
    progress_bar = tqdm(range(episodes))
    if record is None:
        record = False 
    else :
        record = "Recording-{}-vs-{}.mp4".format(record,agent.name, fixed_opponent.name)
    for e in progress_bar:
        trajectory = simultaneous_action_rl_loop.run_episode(env, agent_vector, training=training, record=record)
        trajectories.append(trajectory)
        avg_trajectory_reward = sum(map(lambda experience: experience[2][0], trajectory)) / len(trajectory)
        progress_bar.set_description(f'{mode} {agent.name} against {fixed_opponent.name}. Last avg reward: {avg_trajectory_reward}')
    return trajectories


def learns_against_fixed_opponent_RoboSumo_parallel(agent, fixed_opponent, total_episodes, training_percentage, reward_threshold_percentage, envname='RoboschoolSumo-v0', nbr_parallel_env=2, save=False):
    '''
    Test used to make sure that agent is 'learning' by learning a best response
    against an agent that only plays randomly.
    i.e from random, learns to play only (or mostly) paper
    '''
    maximum_average_reward = 20.0

    training_episodes = int(total_episodes * training_percentage)
    inference_episodes = total_episodes - training_episodes

    env_creator = EnvironmentCreator(envname)
    env = ParallelEnv(env_creator, nbr_parallel_env)
    training_trajectories = simulate_parallel(env, agent, fixed_opponent, episodes=training_episodes, training=True)
    env.close()

    agent.training = False
    if save:
        agent_hook = AgentHook(agent.clone(), save_path='/tmp/test_{}_{}.agent'.format(agent.name, envname))
    
    env = ParallelEnv(env_creator, 1)
    inference_trajectories = simulate_parallel(env, agent, fixed_opponent, episodes=inference_episodes, training=False)
    env.close()

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
