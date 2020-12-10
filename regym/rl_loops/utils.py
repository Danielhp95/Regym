from typing import List, Tuple, Dict, Optional, Any
from copy import deepcopy

from regym.rl_algorithms.agents import Agent
from regym.rl_loops.trajectory import Trajectory


def update_trajectories(trajectories: List[Trajectory],
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
        if extra_infos and i in extra_infos:  # Might not be present
            extra_info = extra_infos[i]
        else: extra_info = None

        acting_agents = current_players[i] if (
            current_players is not None) else None
        trajectories[i].add_timestep(
            o=obs[i], a=action_vector[i], r=rewards[i], succ_o=succ_obs[i],
            done=dones[i], acting_agents=[acting_agents],
            extra_info=extra_info)


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
    if store_extra_information:
        extra_infos = compose_extra_infos(current_players, agent_vector, obs)
    else:
        extra_infos = {env_i: {} for env_i in range(len(current_players))}

    update_trajectories(trajectories, action_vector, res_obs, rewards,
                        res_succ_obs, dones, current_players, extra_infos)


def restructure_parallel_observations(observations: List,
                                      succ_observations: List,
                                      num_players: int) -> List:
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


def compose_extra_infos(current_players: List[int],
                        agent_vector: List[Agent],
                        observations: List) \
                        -> Dict[int, Dict]:
    ''' TODO '''
    environments_per_agent = compute_environments_per_agent(current_players)
    extra_infos = extract_current_predictions(
        current_players, agent_vector, environments_per_agent)
    add_observations_to_extra_infos(
        extra_infos, current_players, observations)
    return extra_infos


def add_observations_to_extra_infos(extra_infos: Dict[int, Dict],
                                    current_players: List[int],
                                    observations: List):
    '''
    Adds observation (state) received by the agent that acted on that observation.

    This is slightly unnecessary, because these observations are the same as
    the ones stored inside of the episode trajectory. However, opponent
    observations might need to be passed onto an agent that has flag
    `requires_opponents_prediction` turned on (For instance, to model opponents
    in sequential games, as explored in the paper
    'On opponent modelling in Expert Iteration')
    '''
    for env_i in extra_infos.keys():
        a_i = current_players[env_i]
        assert 's' not in extra_infos[env_i][a_i], ('This key is reserved to '
                                                    'store observations (States)')
        extra_infos[env_i][a_i]['s'] = observations[a_i][env_i]


def extract_current_predictions(current_players: List[int],
                                agent_vector: List[Agent],
                                environments_per_agent: Dict[int, List[int]]) \
                        -> Dict[int, Dict[str, Any]]:
    '''
    ASSUMES: The batch dimension in `agent.current_prediction` corresponds to
             env_ids in ASCENDING ORDER

    TODO: Perhaps an agents first instead of environments first could deliver
          performance. Bottleneck is still forward / backward passes in neural
          nets. So This doesn't meaningfully matter.
    '''
    # Required for doing the indexing
    return {env_i:
            {
                a_i: parse_individual_entry_in_prediction(
                    agent_vector[a_i],
                    environments_per_agent[a_i].index(env_i)
                )
            }
            for env_i, a_i in enumerate(current_players)}


def compute_environments_per_agent(current_players) -> Dict[int, List[int]]:
    ''' Dicionary where keys are agents and values are
        the environments where they have just acted '''
    environments_per_agent = {
        a_i: [env_id
              for env_id, player_i in enumerate(current_players)
              if a_i == player_i]
        for a_i in set(current_players)
    }
    return environments_per_agent


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


def update_finished_trajectories(ongoing_trajectories: List[Trajectory],
                                 finished_trajectories: List[Trajectory],
                                 done_envs: List[int]) \
        -> Tuple[List[Trajectory], List[Trajectory]]:
    '''Copies finished :param: ongoing_trajectories into
    :param: finished_trajectories as dictated by :param: done_envs'''
    for env_i in done_envs:
        finished_trajectories += [deepcopy(ongoing_trajectories[env_i])]
        ongoing_trajectories[env_i] = Trajectory(
            env_type=ongoing_trajectories[env_i].env_type,
            num_agents=ongoing_trajectories[env_i].num_agents)
    return ongoing_trajectories, finished_trajectories


def agents_to_update_finished_trajectory_sequential_env(trajectory_length: int,
                                                        num_agents: int)\
        -> List[int]:
    '''Computes list of agent indices that need to be updated'''
    agent_indices = list(range(num_agents))
    # This agent already processed the last experience
    agent_indices.pop(trajectory_length % num_agents)
    return agent_indices


def extract_latest_experience_sequential_trajectory(agent_id: int,
                                                    trajectory: Trajectory,
                                                    extract_extra_info: bool = False) -> Tuple:
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

    # Perhaps there's a better way of computting agent's reward?
    # Such as summing up all intermediate rewards for :agent_id:
    # between t1.t and t2.t?
    r = sum(map(lambda t: t.reward[agent_id], trajectory[t1.t:(t2.t + 1)]))
    done = t2.done

    if extract_extra_info:
        extra_info = extract_extra_info_from_sequential_trajectory(agent_id, trajectory)
    else: extra_info = {}
    return (o, a, r, succ_o, done, extra_info)


def extract_extra_info_from_sequential_trajectory(agent_id: int,
                                       trajectory: Trajectory) \
                            -> Dict[int, Dict[str, Any]]:
    '''
    TODO: mention about stiching together current predicions from trajectory
    '''
    time_index_after_agent_action = -(trajectory.num_agents) + 1
    relevant_timesteps: List = trajectory[time_index_after_agent_action:]


    extra_info = {}
    last_acting_timestep = trajectory.last_acting_timestep_for_agent(agent_id)

    # Some algorithms need to fetch info from last things they did
    extra_info['self'] = last_acting_timestep.extra_info[agent_id]

    # Add extra_info from all other agents
    for timestep in relevant_timesteps:
        for a_i, v in timestep.extra_info.items():
            if a_i == agent_id:
                # An agent should not be fed info about it's own predictions
                continue
            assert a_i not in extra_info, 'Breaking cyclic turn assumption'
            extra_info[a_i] = v
    return extra_info
