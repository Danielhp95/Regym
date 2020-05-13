from copy import deepcopy


def update_trajectories(trajectories, action_vector, obs,
                        rewards, succ_obs, dones):
    num_envs = len(trajectories)
    for i in range(num_envs):
        e = (obs[i], action_vector[i], rewards[i], succ_obs[i], dones[i])
        trajectories[i] += [e]


def update_parallel_sequential_trajectories(trajectories,
                                            current_players,
                                            action_vector,
                                            obs,
                                            rewards,
                                            succ_obs, dones):
    num_envs = len(trajectories)
    res_obs, res_succ_obs = restructure_parallel_observations(obs, succ_obs,
                                                              num_players=len(obs))
    for e_i in range(num_envs):
        experience = (res_obs[e_i], action_vector[e_i],
                      rewards[e_i], res_succ_obs[e_i], dones[e_i])
        trajectories[e_i].append(experience)


def restructure_parallel_observations(observations, succ_observations,
                                      num_players):
    '''
    Observations are structured thus:
        - obs[player_index][env_index]
    So they need to be restructured to:
        - obs[env_index][player_index]

    TODO: Make this restructurin gon RegymAsyncVectorEnv to make
          your life easier (i.e, simplifies sequential_rl_loop significantly)
    '''
    num_envs = len(observations[0])
    res_obs = [[observations[player_i][e_i] for player_i in range(num_players)]
               for e_i in range(num_envs)]
    res_succ_obs = [[succ_observations[player_i][e_i] for player_i in range(num_players)]
                    for e_i in range(num_envs)]
    return res_obs, res_succ_obs


def update_finished_trajectories(ongoing_trajectories,
                                 finished_trajectories, dones):
    done_envs = lambda dones: [i for i in range(len(dones)) if dones[i]]
    finished_envs = done_envs(dones)
    for env_i in finished_envs:
        finished_trajectories += [deepcopy(ongoing_trajectories[env_i])]
        ongoing_trajectories[env_i].clear()
    return finished_envs
