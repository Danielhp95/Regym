from typing import List, Tuple, Dict, Optional, Any
from copy import deepcopy

from regym.rl_algorithms.agents import Agent
from regym.rl_loops.trajectory import Trajectory


def update_trajectories(trajectories: List[List],
                        action_vector: List[int], obs: List,
                        rewards: List[float], succ_obs: List,
                        dones: List[bool],
                        current_players: List[int] = None,
                        extra_infos: Dict[int, Optional[Dict[str, Any]]] = None):
    '''
    Appends to each trajectory in :param: trajectories its most recent
    experience (state, action, reward, succ_state, done_flag)
    '''
    num_envs = len(trajectories)
    for i in range(num_envs):
        acting_agents = current_players[i] if (current_players is not None) else None
        trajectories[i].add_timestep(
            o=obs[i], a=action_vector[i], r=rewards[i], succ_o=succ_obs[i],
            done=dones[i], acting_agents=[acting_agents])


def update_parallel_sequential_trajectories(trajectories: List[List],
                                            agent_vector: List[Agent],
                                            action_vector: List[int],
                                            obs: List,
                                            rewards: List[float],
                                            succ_obs: List, dones: List[bool],
                                            current_players: List[int],
                                            store_extra_information: bool):
    '''
    TODO
    :param trajectories:
    :param agent_vector:
    :param action_vector:
    :param obs:
    :param rewards:
    :param succ_obs:
    :param current_players: Mapping of agent indexes that just acted on each environment
    :param store_extra_information:
    '''
    res_obs, res_succ_obs = restructure_parallel_observations(obs, succ_obs,
                                                              num_players=len(obs))
    extra_infos = extract_current_predictions(current_players, agent_vector)\
        if store_extra_information else {}

    update_trajectories(trajectories, action_vector, res_obs, rewards,
                        res_succ_obs, dones, current_players, extra_infos)


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


def update_finished_trajectories(ongoing_trajectories: List[List[Tuple]],
                                 finished_trajectories: List[List[Tuple]],
                                 done_envs: List[int]) \
                                 -> Tuple[List[List[Tuple]], List[List[Tuple]]]:
    '''Copies finished :param: ongoing_trajectories into
    :param: finished_trajectories as dictated by :param: done_envs'''
    for env_i in done_envs:
        finished_trajectories += [deepcopy(ongoing_trajectories[env_i])]
        ongoing_trajectories[env_i] = Trajectory(env_type=finished_trajectories[-1].env_type)
    return ongoing_trajectories, finished_trajectories


def agents_to_update_finished_trajectory_sequential_env(trajectory_length: int,
                                                        num_agents: int)\
                                                        -> List[int]:
    '''Computes list of agent indices that need to be updated'''
    agent_indices = list(range(num_agents))
    # This agent already processed the last experience
    agent_indices.pop(trajectory_length % num_agents)
    return agent_indices
