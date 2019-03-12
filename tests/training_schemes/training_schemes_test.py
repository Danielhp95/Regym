import os
import sys

from tqdm import tqdm


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
    
    '''
    agent_menagerie_path = '{}/{}-{}'.format(menagerie_path, self_play_scheme.name, training_agent.name)
    if not os.path.exists(menagerie_path):
        os.mkdir(menagerie_path)
    if not os.path.exists(agent_menagerie_path):
        os.mkdir(agent_menagerie_path)
    '''

    '''
    menagerie = menagerie
    '''
    trajectories = []
    progress_bar = tqdm(range(target_episodes) )
    for episode in progress_bar:
    #for episode in range(target_episodes):
        if episode % opci == 0:
            opponent_agent_vector_e = self_play_scheme.opponent_sampling_distribution(menagerie, training_agent)
        
        '''
        if isinstance(env, ParallelEnv):
            episode_trajectory = run_episode_parallel(env, [training_agent]+opponent_agent_vector_e, training=True)
        else:
            episode_trajectory = run_episode(env, [training_agent]+opponent_agent_vector_e, training=True)
        '''
        episode_trajectory = None 
        '''
        candidate_save_path = f'{agent_menagerie_path}/checkpoint_episode_{iteration + episode}.pt'
        '''
        candidate_save_path = "/tmp/agent.agent"
        menagerie = self_play_scheme.curator(menagerie, training_agent, episode_trajectory, candidate_save_path=candidate_save_path)
        
        '''
        trajectories.append(episode_trajectory)
        '''
        progress_bar.set_description(f"Training process {self_play_scheme.name} :: {training_agent.name} :: episode : {episode}/{target_episodes}")
    
    return menagerie, training_agent, trajectories
