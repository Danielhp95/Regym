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
    num_envs = range(len(trajectories))
    for e_i, p_i in enumerate(current_players):
        e = (obs[p_i][e_i], action_vector[e_i],
             rewards[e_i][p_i], succ_obs[p_i][e_i], dones[e_i])
        trajectories[e_i] += [e]


def update_finished_trajectories(ongoing_trajectories,
                                 finished_trajectories, dones):
    done_envs = lambda dones: [i for i in range(len(dones)) if dones[i]]
    finished_envs = done_envs(dones)
    for env_i in finished_envs:
        finished_trajectories += [ongoing_trajectories[env_i]]
        ongoing_trajectories[env_i] = []
    return finished_envs
