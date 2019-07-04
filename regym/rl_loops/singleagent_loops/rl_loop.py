import math
from tqdm import tqdm

def run_episode(env, agent, training, max_episode_length=math.inf):
    '''
    Runs a single episode of a single-agent rl loop until termination.
    :param env: OpenAI gym environment
    :param agent: Agent policy used to take actions in the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param max_episode_length: Maximum expisode duration meassured in steps.
    :returns: Episode trajectory. list of (o,a,r,o')
    '''
    observation = env.reset()
    done = False
    trajectory = []
    for step in tqdm(range(max_episode_length)):
        action = agent.take_action(observation)
        succ_observation, reward, done, info = env.step(action)
        trajectory.append((observation, action, reward, succ_observation, done))
        if training: agent.handle_experience(observation, action, reward, succ_observation, done)
        observation = succ_observation
        if done:
            break

    return trajectory
