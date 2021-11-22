from enum import Enum
from typing import List, Callable, Tuple, Any, Optional
import os
from os import listdir
from os.path import isfile, join

import torch
import tqdm
from .agents import TabularQLearningAgent, DeepQNetworkAgent, PPOAgent, MixedStrategyAgent

AgentType = Enum("AgentType", "DQN TQL PPO MixedStrategyAgent")


# TODO: move elsewhere. Maybe utils?
# TODO: test function
def load_population_from_path(path: str, file_extension='pt',
                              sort_fn: Callable[[str], Tuple[Any]]=None,
                              show_progress: bool=False,
                              state_preprocess_fn: Callable=Optional[None]) -> List:
    '''
    Loads a population of agents from :param: path.
    Agent files are recognized by the :param: file_extension.
    If :param: sort_fn is passed, all appropiate files in :param: path
    are sorted according to :param: sort_fn.

     NOTE: it might be easier to sort agents once they are loaded
     (for instance, by looking at the number of finished_episodes).
     :param: sort_fn is used to sort over file paths (i.e strings)

    :param path: Relative path from which
    :param file_extension: Only files with this extension will be loaded
    :param sort_fn: Function to be used as part of list.sort(key={}), to
                    sort agent files before loading.
    :param state_preprocess_fn: Function to replace the loaded population's state_preprocess_fn
    :returns: Loaded population from :param: path with potentially updated state_preprocess_fn
    '''
    files = [os.path.abspath(f'{path}/{f}') for f in listdir(path)
             if isfile(join(path, f)) and f.endswith(file_extension)]
    if sort_fn is not None: files.sort(key=sort_fn)
    if show_progress: files = tqdm.tqdm(files, desc=f'Loading agents from {path}')
    loaded_population = [torch.load(open(f, 'rb')) for f in files]
    if state_preprocess_fn:
        for agent in loaded_population: agent.state_preprocess_fn = state_preprocess_fn
    return loaded_population


class AgentHook():
    def __init__(self, agent, save_path=None):
        """
        Creates an agent hook which allows to transport :param: agent:
        - Between processes if by making all Torch.Tensors be in CPU IF :param: save_path is None
        - Written to disk if at path :param: save_path if it is not None

        :param agent: Agent to be hooked to be transported between processes
        :param save_path: path where to save the current agent.
        :returns: AgentHook agent whose type is that of :param: agent
        """

        self.name = agent.name
        self.save_path = save_path

        if isinstance(agent, MixedStrategyAgent):
            agent_type, model_list = AgentType.MixedStrategyAgent, []
        elif isinstance(agent, TabularQLearningAgent):
            agent_type, model_list = AgentType.TQL, []
        elif isinstance(agent, DeepQNetworkAgent):
            agent_type, model_list = AgentType.DQN, [('model', agent.algorithm.model), ('target_model', agent.algorithm.target_model)]
        elif isinstance(agent, PPOAgent):
            agent_type, model_list = AgentType.PPO, [('model', agent.algorithm.model)]
        self.hook_agent(agent, agent_type, model_list)

    def hook_agent(self, agent, agent_type, model_list):
        self.type, self.model_list = agent_type, model_list
        for _, model in model_list: model.cpu()
        if not self.save_path: self.agent = agent
        else: torch.save(agent, self.save_path)

    @staticmethod
    def unhook(agent_hook, use_cuda=None):
        if hasattr(agent_hook, 'save_path') and agent_hook.save_path is not None: agent_hook.agent = torch.load(agent_hook.save_path)
        if agent_hook.type == AgentType.TQL or agent_hook.type == AgentType.MixedStrategyAgent: return agent_hook.agent
        if 'use_cuda' in agent_hook.agent.algorithm.kwargs:
            if use_cuda is not None:
                agent_hook.agent.algorithm.kwargs['use_cuda'] = use_cuda
                if hasattr(agent_hook.agent, 'state_preprocessing'): agent_hook.agent.state_preprocessing.use_cuda = use_cuda
            if agent_hook.agent.algorithm.kwargs['use_cuda']:
                for name, model in agent_hook.model_list: setattr(agent_hook.agent.algorithm, name, model.cuda())
        return agent_hook.agent
