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
    generator = tqdm(range(int(max_episode_length))) if max_episode_length != math.inf else range(int(1e20))
    for step in generator:
        action = agent.take_action(observation)
        succ_observation, reward, done, info = env.step(action)
        trajectory.append((observation, action, reward, succ_observation, done))
        if training: agent.handle_experience(observation, action, reward, succ_observation, done)
        observation = succ_observation
        if done:
            break

    return trajectory

def run_episode_parallel(env, agent, training):
    '''
    Runs a single multi-agent rl loop until termination.
    The observations vector is of length n, where n is the number of agents
    observations[i] corresponds to the oberservation of agent i.
    :param env: ParallelEnv wrapper around an OpenAI gym environment
    :param agent: Agent policy used to take actionsin the environment and to process simulated experiences
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :returns: Trajectory (o,a,r,o')
    '''
    observations = env.reset()
    nbr_actors = env.get_nbr_envs()
    done = [False]*nbr_actors
    previous_done = copy.deepcopy(done)

    per_actor_trajectories = {i:list() for i in range(nbr_actors)}
    trajectory = []
    while not all(done):
        action = agent.take_action(observations)
        succ_observations, reward_vector, done, info = env.step(action)


        batch_index = -1
        for actor_index in per_actor_trajectories.keys():
            if done[actor_index] and previous_done[actor_index]:
                continue
            batch_index +=1

            pa_obs = observations[batch_index]
            pa_a = action[batch_index]
            pa_r = reward[batch_index]
            pa_succ_obs = succ_observations[batch_index]
            pa_done = done[actor_index]
            per_actor_trajectories[actor_index].append( (pa_obs, pa_a, pa_r, pa_succ_obs, pa_done) )

            if actor_index == 0 :
                trajectory.append( (pa_obs, pa_a, pa_r, pa_succ_obs, done[actor_index]) )

        observations = copy.deepcopy(succ_observations)
        previous_done = copy.deepcopy(done)

    if training:
        # Let us handle the experience (actor-)sequence by (actor-)sequence: 
        for actor_index in per_actor_trajectories.keys():
            for pa_obs, pa_a, pa_r, pa_succ_obs, pa_done in per_actor_trajectories[actor_index]:
                    agent.handle_experience( pa_obs.reshape((1,-1)), pa_a.reshape((1,-1)), pa_r.reshape((1,-1)), pa_succ_obs.reshape((1,-1)), pa_done)

    return per_actor_trajectories