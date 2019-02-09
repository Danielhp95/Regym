import os


def run_episode(env, agent_vector, training):
    '''
    Runs a single multi-agent rl loop until termination.
    The observations vector is of length n, where n is the number of agents
    observations[i] corresponds to the oberservation of agent i.
    :param env: OpenAI gym environment
    :param agent_vector: Vector containing the agent for each agent in the environment
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :returns: Trajectory (o,a,r,o')
    '''
    observations = env.reset()
    done = False
    trajectory = []
    while not done:
        action_vector = [agent.take_action(observations[i]) for i, agent in enumerate(agent_vector)]
        succ_observations, reward_vector, done, info = env.step(action_vector)
        trajectory.append((observations, action_vector, reward_vector, succ_observations, done))
        observations = succ_observations
        if training:
            for i, agent in enumerate(agent_vector):
                agent.handle_experience(observations[i], action_vector[i], reward_vector[i], succ_observations[i], done)

    return trajectory


def self_play_training(env, training_agent, self_play_scheme, target_episodes=10, opci=1, menagerie=[], results_path=None, iteration=None):
    '''
    Extension of the multi-agent rl loop. The extension works thus:
    - Opponent sampling distribution
    - MARL loop
    - Curator

    :param env: OpenAI gym environment
    :param training_scheme
    :param training_agent: AgentHook of the agent being trained, together with training algorithm
    :param opponent_sampling_distribution: Probability distribution that
    :param curator: Gating function which determines if the current agent will be added to the menagerie at the end of an episode
    :param target_episodes: number of episodes that will be run before training ends.
    :param opci: Opponent policy Change Interval
    :param results_path: path of the folder where all results relevant to the current run are being stored.
    :returns: Menagerie after target_episodes have elapsed
    :returns: Trained agent. freshly baked!
    :returns: Array of arrays of trajectories for all target_episodes
    '''
    menagerie_path = '{}/menagerie'.format(results_path)
    agent_menagerie_path = '{}/{}'.format(menagerie_path, training_agent.name)
    if not os.path.exists(menagerie_path):
        os.mkdir(menagerie_path)
        os.mkdir(agent_menagerie_path)

    # Loading the model from the AgentHook: TODO maybe rename agentHook
    training_agentHook = training_agent
    training_agent = training_agent(training=True)

    menagerie = menagerie
    trajectories = []
    for episode in range(target_episodes):
        training_agentHook = training_agent.clone(training=False)
        if episode % opci == 0:
            opponent_agent_vector_e = self_play_scheme.opponent_sampling_distribution(menagerie, training_agentHook)
        episode_trajectory = run_episode(env, [training_agent]+opponent_agent_vector_e, training=True)
        menagerie = self_play_scheme.curator(menagerie, training_agentHook, episode_trajectory)
        trajectories.append(episode_trajectory)

    path = os.path.join(agent_menagerie_path, 'checkpoint_episode_{}.pt'.format(iteration))
    return menagerie, training_agent.clone(training=True, path=path), trajectories
