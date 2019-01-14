import logging

def run_episode(env, policy_vector, training,name=None):
    '''
    Runs a single multi-agent rl loop until termination.
    :param env: OpenAI gym environment
    :param policy_vector: Vector containing the policy for each agent in the environment
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :returns: Trajectory (s,a,r,s')
    '''
    if name is None :
        name = "None"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    #logger.info('Reset environment :: ...')
    state = env.reset()
    #logger.info('Reset environment :: OK.')
    done = False
    trajectory = []
    it = 0 
    while not done:
        it +=1 
        #logger.info("Iteration : {} ".format(it))
        action_vector = [agent.take_action(state) for agent in policy_vector]
        succ_state, reward_vector, done, info = env.step(action_vector)
        trajectory.append((state, action_vector, reward_vector, succ_state, done))
        if training:
            for i, agent in enumerate(policy_vector):
                agent.handle_experience(state, action_vector[i], reward_vector[i], succ_state, done)
        
    return trajectory


def self_play_training(env, training_policy, self_play_scheme, target_episodes=10, opci=1, menagerie=[],name=None):
    '''
    Extension of the multi-agent rl loop. The extension works thus:
    - Opponent sampling distribution
    - MARL loop
    - Curator

    :param env: OpenAI gym environment
    :param training_scheme
    :param training_policy: policy being trained, together with training algorithm
    :param opponent_sampling_distribution: Probability distribution that
    :param curator: Gating function which determines if the current policy will be added to the menagerie at the end of an episode
    :param target_episodes: number of episodes that will be run before training ends.
    :param opci: Opponent Policy Change Interval
    :returns: Menagerie after target_episodes have elapsed
    :returns: Trained policy. freshly baked!
    :returns: Array of arrays of trajectories for all target_episodes
    '''
    if name is None :
        name = "None"
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    menagerie = menagerie
    trajectories = []
    for episode in range(target_episodes):
        if episode % opci == 0:
            #logger.info("Running episode : {} / {} :: choosing opponent :: ...".format(episode,target_episodes) )
            opponent_policy_vector_e = self_play_scheme.opponent_sampling_distribution(menagerie, training_policy.clone(training=False))
            #logger.info("Running episode : {} / {} :: choosing opponent :: OK.".format(episode,target_episodes) )
        #logger.info("Running episode : {} / {} :: ...".format(episode,target_episodes) )
        episode_trajectory = run_episode(env, [training_policy]+opponent_policy_vector_e, training=True,name=name)
        #logger.info("Running episode : {} / {} :: OK.".format(episode,target_episodes) )
        menagerie = self_play_scheme.curator(menagerie, training_policy.clone(training=False), episode_trajectory)
        trajectories.append(episode_trajectory)
        
    return menagerie, training_policy.clone(training=True), trajectories
