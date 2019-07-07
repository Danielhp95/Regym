import os
import copy
import gym
from OpenGL import GL
from tqdm import tqdm
import numpy as np

episode_n = 0

def run_episode(env, agent_vector, training, record=False):
    '''
    Runs a single multi-agent rl loop until termination.
    The observations vector is of length n, where n is the number of agents
    observations[i] corresponds to the oberservation of agent i.
    :param env: OpenAI gym environment
    :param agent_vector: Vector containing the agent for each agent in the environment
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :returns: Episode trajectory (o,a,r,o')
    '''
    if record:
        global episode_n
        episode_n +=1
        video_recorder = gym.wrappers.monitoring.video_recorder.VideoRecorder(env=env, base_path=("/tmp/{}-episode-{}".format(record, episode_n)), enabled=True)

    observations = copy.deepcopy( env.reset() )
    done = False
    trajectory = []
    timestep = 0

    inner_loop = True

    while not done:
        action_vector = copy.deepcopy( [agent.take_action(observations[i]) for i, agent in enumerate(agent_vector)] )
        succ_observations, reward_vector, done, info = copy.deepcopy( env.step(action_vector) )
        trajectory.append( copy.deepcopy( (observations, action_vector, reward_vector, succ_observations, done) ) )

        if inner_loop:
            if training:
                for i, agent in enumerate(agent_vector):
                    agent.handle_experience( copy.deepcopy(observations[i]), copy.deepcopy(action_vector[i]), copy.deepcopy(reward_vector[i]), copy.deepcopy(succ_observations[i]), copy.deepcopy(done) )

        if record:
            video_recorder.capture_frame()

        timestep += 1
        observations = copy.deepcopy(succ_observations)


    if not(inner_loop) and training:
        for o, a, r, so, d in trajectory:
            for i, agent in enumerate(agent_vector):
                #a_dummy = agent.take_action(o[i])
                #agent.handle_experience(o[i], a_dummy, r[i], so[i], d)
                agent.handle_experience(o[i], a[i], r[i], so[i], d)

    if record:
        video_recorder.close()
        print("Video recorded :: episode {}".format(episode_n))

    return trajectory

def run_episode_parallel(env, agent_vector, training, self_play=True):
    '''
    Runs a single multi-agent rl loop until termination.
    The observations vector is of length n, where n is the number of agents
    observations[i] corresponds to the oberservation of agent i.
    :param env: ParallelEnv wrapper around an OpenAI gym environment
    :param agent_vector: Vector containing the agent for each agent in the environment
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :param self_play: boolean specifying the mode in which to operate and, thus, what to return
    :returns: Trajectory (o,a,r,o') of actor 0, if self_play, trajectories for all the actor otherwise.
    '''
    nbr_actors = env.get_nbr_envs()
    observations = env.reset()
    for agent in agent_vector: 
        agent.set_nbr_actor(nbr_actors)
        agent.reset_actors()
    done = [False]*nbr_actors
    previous_done = copy.deepcopy(done)

    per_actor_trajectories = [list() for i in range(nbr_actors)]
    trajectory = []
    while not all(done):
        action_vector = [agent.take_action(observations[i]) for i, agent in enumerate(agent_vector)]
        succ_observations, reward_vector, done, info = env.step(action_vector)

        if training:
            for i, agent in enumerate(agent_vector):
                agent.handle_experience(observations[i], 
                                        action_vector[i], 
                                        reward_vector[i], 
                                        succ_observations[i], 
                                        done)
        
        batch_index = -1
        batch_idx_done_actors_among_not_done = []
        for actor_index in range(nbr_actors):
            if previous_done[actor_index]:
                continue
            batch_index +=1
            
            # Bookkeeping of the actors whose episode just ended:
            if done[actor_index] and not(previous_done[actor_index]):
                batch_idx_done_actors_among_not_done.append(batch_index)
                
            pa_obs = [ observations[idx_agent][batch_index] for idx_agent, agent in enumerate(agent_vector) ]
            pa_a = [ action_vector[idx_agent][batch_index] for idx_agent, agent in enumerate(agent_vector) ]
            pa_r = [ reward_vector[idx_agent][batch_index] for idx_agent, agent in enumerate(agent_vector) ]
            pa_succ_obs = [ succ_observations[idx_agent][batch_index] for idx_agent, agent in enumerate(agent_vector) ]
            pa_done = [ done[actor_index] for idx_agent, agent in enumerate(agent_vector) ]
            per_actor_trajectories[actor_index].append((pa_obs, pa_a, pa_r, pa_succ_obs, pa_done))

            if actor_index == 0 :
                trajectory.append((pa_obs, pa_a, pa_r, pa_succ_obs, done[actor_index]))

        observations = copy.deepcopy(succ_observations)
        if len(batch_idx_done_actors_among_not_done):
            # Regularization of the agents' next observations:
            batch_idx_done_actors_among_not_done.sort(reverse=True)
            for i,agent in enumerate(agent_vector):
                for batch_idx in batch_idx_done_actors_among_not_done:
                    observations[i] = np.concatenate( [observations[i][:batch_idx,...], observations[i][batch_idx+1:,...]], axis=0)
        
        previous_done = copy.deepcopy(done)

    if self_play:
        return trajectory

    return per_actor_trajectories


def self_play_training(env, training_agent, self_play_scheme, target_episodes=10, opci=1, menagerie=[], menagerie_path=None, iteration=None):
    '''
    Extension of the multi-agent rl loop. The extension works thus:
    - Opponent sampling distribution
    - MARL loop
    - Curator

    :param env: ParallelEnv wrapper around an OpenAI gym environment
    :param self_play_scheme: Self-play scheme used to train the agent
    :param training_agent: AgentHook of the agent being trained, together with training algorithm
    :param opponent_sampling_distribution: Probability distribution that
    :param curator: Gating function which determines if the current agent will be added to the menagerie at the end of an episode
    :param target_episodes: number of episodes that will be run before training ends.
    :param opci: Opponent policy Change Interval
    :param menageries_path: path to folder where all menageries are stored.
    :returns: Menagerie after target_episodes have elapsed
    :returns: Trained agent. freshly baked!
    :returns: Array of arrays of trajectories for all target_episodes
    '''
    agent_menagerie_path = '{}/{}-{}'.format(menagerie_path, self_play_scheme.name, training_agent.name)
    if not os.path.exists(agent_menagerie_path):
        os.mkdir(agent_menagerie_path)

    trajectories = []
    progress_bar = tqdm(range(target_episodes) )
    for episode in progress_bar:
    #for episode in range(target_episodes):
        if episode % opci == 0:
            opponent_agent_vector_e = self_play_scheme.opponent_sampling_distribution(menagerie, training_agent)
        if isinstance(env, ParallelEnv):
            episode_trajectory = run_episode_parallel(env, [training_agent]+opponent_agent_vector_e, training=True)
        else:
            episode_trajectory = run_episode(env, [training_agent]+opponent_agent_vector_e, training=True)
        candidate_save_path = f'{agent_menagerie_path}/checkpoint_episode_{iteration + episode}.pt'
        menagerie = self_play_scheme.curator(menagerie, training_agent, episode_trajectory, candidate_save_path=candidate_save_path)
        trajectories.append(episode_trajectory)
        progress_bar.set_description(f"Training process {self_play_scheme.name} :: {training_agent.name} :: episode : {episode}/{target_episodes}")

    return menagerie, training_agent, trajectories
