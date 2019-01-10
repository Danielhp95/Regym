def run_episode(env, policy_vector, training):
    '''
    Runs a single multi-agent rl loop until termination.
    :param env: OpenAI gym environment
    :param policy_vector: Vector containing the policy for each agent in the environment
    :param training: (boolean) Whether the agents will learn from the experience they recieve
    :returns: Trajectory (s,a,r,s')
    '''
    state = env.reset()
    done = False
    trajectory = []
    cum_reward = None
    step = 0
    while not done:
        action_vector = [agent.take_action(state) for agent in policy_vector]
        succ_state, reward_vector, done, info = env.step(action_vector)
        trajectory.append((state, action_vector, reward_vector, succ_state, done))
        if training:
            for i, agent in enumerate(policy_vector):
                agent.handle_experience(state, action_vector[i], reward_vector[i], succ_state, done)
        
        step += 1
        state = succ_state
        if cum_reward is None :
            cum_reward = reward_vector
        else :
            for i in range(len(cum_reward)):
                cum_reward[i] += reward_vector[i]
        print("Step:{} / Cumulative Reward: {} / Actions: {}".format( step, cum_reward, action_vector), end='\r' )

    return trajectory, cum_reward


def self_play_training(env, training_policy, self_play_scheme, target_episodes=10, opci=1, menagerie=[]):
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
    menagerie = menagerie
    trajectories = []
    cum_cum_reward = None 
    for episode in range(target_episodes):
        print("Episode:{}".format(episode))
        if episode % opci == 0:
            opponent_policy_vector_e = self_play_scheme.opponent_sampling_distribution(menagerie, training_policy.clone(training=False))
        episode_trajectory, cum_reward = run_episode(env, [training_policy]+opponent_policy_vector_e, training=True)
        menagerie = self_play_scheme.curator(menagerie, training_policy.clone(training=False), episode_trajectory)
        trajectories.append(episode_trajectory)
        
        print('\n', end='\r')

        if cum_cum_reward is None :
            cum_cum_reward = cum_reward
        else :
            for i in range(len(cum_reward)):
                cum_cum_reward[i] += cum_reward[i]

        print('RUNNING CUM_REWARD: {}', cum_cum_reward)

    return menagerie, training_policy.clone(training=True), trajectories
