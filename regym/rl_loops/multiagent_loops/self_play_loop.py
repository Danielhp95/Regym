from typing import List
import numpy as np
import os

def self_play_training(task, training_agent, self_play_scheme,
                       target_episodes: int=10, opci: int=1,
                       menagerie: List=[], menagerie_path: str=None,
                       iteration: int=None):
    '''
    Extension of the multi-agent rl loop. The extension works thus:
    - Opponent sampling distribution
    - MARL loop
    - Curator

    :param task: Mutiagent task
    :param training_scheme: Self play training scheme that extends the multiagent rl loop
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
    for episode in range(target_episodes):
        if episode % opci == 0:
            opponent_agent_vector_e = self_play_scheme.opponent_sampling_distribution(menagerie, training_agent)
        training_agent_index = np.random.choice(range(len(opponent_agent_vector_e)))
        opponent_agent_vector_e.insert(training_agent_index, training_agent)
        episode_trajectory = task.run_episode(agent_vector=opponent_agent_vector_e, training=True)
        candidate_save_path = f'{agent_menagerie_path}/checkpoint_episode_{iteration + episode}.pt'

        menagerie = self_play_scheme.curator(menagerie, training_agent,
                                             episode_trajectory, training_agent_index,
                                             candidate_save_path=candidate_save_path)
        trajectories.append(episode_trajectory)

    return menagerie, training_agent, trajectories
