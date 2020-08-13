from typing import List, Tuple, Optional
import numpy as np
import os

from tqdm import trange


def self_play_training(task, training_agent, self_play_scheme,
                       target_episodes: int=10, opci: int=1,
                       menagerie: List=[],
                       menagerie_path: str='.',
                       initial_episode: int=0,
                       shuffle_agent_positions: bool = True,
                       agent_position: Optional[int] = None,
                       show_progress: bool = False) \
                       -> Tuple[List, 'Agent', List]:
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
    :param initial_episode: Episode from where training takes on. Useful when training is interrupted.
    :param shuffle_agent_positions: Whether to randomize the position of each
                                    agent in the environment each episode.
    :param agent_position: Optional, fixes the position in the environment
                           of the learning agent during all training.
    :param show_progress: Whether to output a progress bar to stdout
    :returns: Menagerie after target_episodes have elapsed
    :returns: Trained agent. freshly baked!
    :returns: Array of arrays of trajectories for all target_episodes
    '''
    # TODO: when bored, make a check for valid inputs (error handling)
    agent_menagerie_path = '{}/{}-{}'.format(menagerie_path, self_play_scheme.name, training_agent.name)
    if not os.path.exists(agent_menagerie_path):
        os.mkdir(agent_menagerie_path)

    trajectories = []

    if show_progress:
        episodes = trange(target_episodes,
                          desc=f'Self-play training {training_agent.name} under {self_play_scheme.name}:')
    else:
        episodes = range(target_episodes)

    for episode in episodes:

        if shuffle_agent_positions: training_agent_index = np.random.choice(range(task.num_agents))
        else: training_agent_index = agent_position

        if episode % opci == 0:
            opponent_agent_vector_e = self_play_scheme.opponent_sampling_distribution(menagerie, training_agent)
        opponent_agent_vector_e.insert(training_agent_index, training_agent)

        episode_trajectory = task.run_episode(agent_vector=opponent_agent_vector_e, training=True)

        candidate_save_path = f'{agent_menagerie_path}/checkpoint_episode_{initial_episode + episode}.pt'
        menagerie = self_play_scheme.curator(menagerie, training_agent,
                                             episode_trajectory, training_agent_index,
                                             candidate_save_path=candidate_save_path)
        trajectories.append(episode_trajectory)

    return menagerie, training_agent, trajectories
