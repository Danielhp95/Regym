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
        acting_agents = current_players[i] if (
            current_players is not None) else None
        trajectories[i].add_timestep(
            o=obs[i], a=action_vector[i], r=rewards[i], succ_o=succ_obs[i],
            done=dones[i], acting_agents=[acting_agents],
            extra_info=extra_infos[i])


def update_parallel_sequential_trajectories(trajectories: List[Trajectory],
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
        if store_extra_information else {env_i: {} for env_i in range(len(current_players))}

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


def extract_current_predictions(current_players: List[int],
                                agent_vector: List[Agent]) \
                        -> Dict[int, Dict[str, Any]]:
    '''
    ASSUMES: The batch dimension in `agent.current_prediction` corresponds to
             env_ids in ASCENDING ORDER

    TODO: Perhaps an agents first instead of environments first could deliver
          performance. Bottleneck is still forward / backward passes in neural
          nets. So This doesn't meaningfully matter.
    '''
    # Required for doing the indexing
    environments_per_agent = {
        a_i: [env_id
              for env_id, player_i in enumerate(current_players)
              if a_i == player_i]
        for a_i in set(current_players)
    }
    return {env_i:
            {
                a_i: parse_individual_entry_in_prediction(
                    agent_vector[a_i],
                    environments_per_agent[a_i].index(env_i)
                )
            }
            for env_i, a_i in enumerate(current_players)}


def parse_individual_entry_in_prediction(agent: Agent,
                                         value_index: int) -> Dict[str, Any]:
    '''
    ASSUMES:
        - :param: agent has property `agent.current_prediction: Dict[str, Any]`
        - Each entry in agent.current_prediction has shape:
            [env_i, entry] (i.e it's an entry with a batch dimension)

    :param agent: Agent for which an entry in its predictions is going to be created
    :param value_index: Batch index of the agent's predictions (TODO: explain better)
    '''
    if agent.current_prediction is None:
        return None  # Maybe turn this into empty dictionary
    return {key: value[value_index]
            for key, value in agent.current_prediction.items()}


def update_finished_trajectories(ongoing_trajectories: List[List[Tuple]],
                                 finished_trajectories: List[List[Tuple]],
                                 done_envs: List[int]) \
        -> Tuple[List[List[Tuple]], List[List[Tuple]]]:
    '''Copies finished :param: ongoing_trajectories into
    :param: finished_trajectories as dictated by :param: done_envs'''
    for env_i in done_envs:
        finished_trajectories += [deepcopy(ongoing_trajectories[env_i])]
        ongoing_trajectories[env_i] = Trajectory(
            env_type=finished_trajectories[-1].env_type)
    return ongoing_trajectories, finished_trajectories


def agents_to_update_finished_trajectory_sequential_env(trajectory_length: int,
                                                        num_agents: int)\
        -> List[int]:
    '''Computes list of agent indices that need to be updated'''
    agent_indices = list(range(num_agents))
    # This agent already processed the last experience
    agent_indices.pop(trajectory_length % num_agents)
    return agent_indices


def extract_latest_experience_sequential_trajectory(agent_id: int, trajectory: Trajectory) -> Tuple:
    '''
    ASSUMPTION:
        - every non-terminal observation corresponds to
          the an information set unique for the player whose turn it is.
          This means that each "experience" is from which an RL agent will learn
          (o, a, r, o') is fragmented throughout the trajectory. This function
          "stiches together" the right environmental signals, ensuring that
          each agent only has access to information from their own information sets.

    :param agent_id: Index of agent which will receive a new experience
    :param trajectory: Current episode trajectory
    '''
    t1 = trajectory.last_acting_timestep_for_agent(agent_id)
    t2 = trajectory[-1]

    o, succ_o = t1.observation[agent_id], t2.succ_observation[agent_id]
    a = t1.action
    r = t1.reward[agent_id]  # Perhaps there's a better way of computting agent's reward?
                             # Such as summing up all intermediate rewards for :agent_id:
                             # between t1.t and t2.t?
    done = t2.done

    extra_info = extract_extra_info_from_sequential_trajectory(agent_id, trajectory)
    return (o, a, r, succ_o, done, extra_info)


def extract_extra_info_from_sequential_trajectory(agent_id: int,
                                       trajectory: Trajectory) \
                            -> Dict[int, Dict[str, Any]]:
    '''
    TODO: mention about stiching together current predicions from trajectory
    '''
    time_index_after_agent_action = -(trajectory.num_agents) + 1
    relevant_timesteps = trajectory[time_index_after_agent_action:]

    extra_info = {}
    for timestep in relevant_timesteps:
        for a_i, v in timestep.extra_info.items():
            if a_i == agent_id:
                # An agent should not be fed info about it's own predictions
                continue
            assert a_i not in extra_info, 'Breaking cyclic turn assumption'
            extra_info[a_i] = v
    return extra_info
