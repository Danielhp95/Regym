import os
import numpy as np

from regym.rl_algorithms.agents import build_PPO_Agent
from regym.rl_algorithms.agents import build_DQN_Agent
from regym.rl_algorithms.agents import build_TabularQ_Agent
from regym.rl_algorithms.agent_hook import AgentHook, AgentType

from test_fixtures import RPSTask
from test_fixtures import ppo_config_dict, dqn_config_dict, tabular_q_learning_config_dict


def test_can_hook_tql_agent(RPSTask, tabular_q_learning_config_dict):
    agent = build_TabularQ_Agent(RPSTask, tabular_q_learning_config_dict, 'TQL')
    hook = AgentHook(agent)

    compare_against_expected_agenthook(agent, hook, AgentType.TQL, [])


def test_can_hook_dqn_agent_using_cuda(RPSTask, dqn_config_dict):
    dqn_config_dict['use_cuda'] = True
    agent = build_DQN_Agent(RPSTask, dqn_config_dict, 'DQN')
    assert all(map(lambda param: param.is_cuda, agent.algorithm.model.parameters()))
    assert all(map(lambda param: param.is_cuda, agent.algorithm.target_model.parameters()))
    hook = AgentHook(agent)

    compare_against_expected_agenthook(agent, hook, AgentType.DQN, [hook.agent.algorithm.model, hook.agent.algorithm.target_model])


def test_can_hook_ppo_agent_using_cuda(RPSTask, ppo_config_dict):
    ppo_config_dict['use_cuda'] = True
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    assert all(map(lambda param: param.is_cuda, agent.algorithm.model.parameters()))
    hook = AgentHook(agent)

    compare_against_expected_agenthook(agent, hook, AgentType.PPO, [hook.agent.algorithm.model])


def compare_against_expected_agenthook(agent, hooked_agent, expected_hook_type, model_list):
    assert hooked_agent.type == expected_hook_type
    assert hooked_agent.agent == agent
    for model in model_list: assert all(map(lambda param: not param.is_cuda, hooked_agent.agent.algorithm.model.parameters()))


def test_can_unhook_tql_agenthook(RPSTask, tabular_q_learning_config_dict):
    agent = build_TabularQ_Agent(RPSTask, tabular_q_learning_config_dict, 'TQL')
    hook = AgentHook(agent)
    retrieved_agent = AgentHook.unhook(hook)

    compare_against_expected_retrieved_agent(agent, retrieved_agent, [])


def test_can_unhook_dqn_agenthook_cuda(RPSTask, dqn_config_dict):
    dqn_config_dict['use_cuda'] = True
    agent = build_DQN_Agent(RPSTask, dqn_config_dict, 'DQN')
    assert all(map(lambda param: param.is_cuda, agent.algorithm.model.parameters()))
    assert all(map(lambda param: param.is_cuda, agent.algorithm.target_model.parameters()))
    hook = AgentHook(agent)
    retrieved_agent = AgentHook.unhook(hook)

    compare_against_expected_retrieved_agent(agent, retrieved_agent, [retrieved_agent.algorithm.model, retrieved_agent.algorithm.target_model])


def test_can_unhook_ppo_agenthook_with_cuda(RPSTask, ppo_config_dict):
    ppo_config_dict['use_cuda'] = True
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    assert all(map(lambda param: param.is_cuda, agent.algorithm.model.parameters()))
    hook = AgentHook(agent)

    retrieved_agent = AgentHook.unhook(hook)

    compare_against_expected_retrieved_agent(agent, retrieved_agent, [retrieved_agent.algorithm.model])


def compare_against_expected_retrieved_agent(agent, retrieved_agent, model_list):
    assert agent == retrieved_agent
    assert_model_parameters_are_cuda_tensors(model_list)


def test_can_save_tql_to_memory(RPSTask, tabular_q_learning_config_dict):
    agent = build_TabularQ_Agent(RPSTask, tabular_q_learning_config_dict, 'TQL')
    save_path = '/tmp/test_save.agent'
    hook = AgentHook(agent, save_path)

    assess_file_has_been_saved_on_disk_and_not_on_ram(hook, save_path)
    os.remove(save_path)


def test_can_save_dqn_to_memory(RPSTask, dqn_config_dict):
    agent = build_DQN_Agent(RPSTask, dqn_config_dict, 'DQN')
    save_path = '/tmp/test_save.agent'
    hook = AgentHook(agent, save_path)

    assess_file_has_been_saved_on_disk_and_not_on_ram(hook, save_path)
    os.remove(save_path)


def test_can_save_ppo_to_memory(RPSTask, ppo_config_dict):
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    save_path = '/tmp/test_save.agent'
    hook = AgentHook(agent, save_path=save_path)

    assess_file_has_been_saved_on_disk_and_not_on_ram(hook, save_path)
    os.remove(save_path)


def test_can_load_tql_from_agenthook(RPSTask, tabular_q_learning_config_dict):
    agent = build_TabularQ_Agent(RPSTask, tabular_q_learning_config_dict, 'TQL')
    save_path = '/tmp/test_save.agent'
    hook = AgentHook(agent, save_path=save_path)

    retrieved_agent = AgentHook.unhook(hook)
    assert np.array_equal(agent.algorithm.Q_table, retrieved_agent.algorithm.Q_table)


def test_can_load_dqn_from_agenthook_with_cuda(RPSTask, dqn_config_dict):
    dqn_config_dict['use_cuda'] = True
    agent = build_DQN_Agent(RPSTask, dqn_config_dict, 'DQN')
    save_path = '/tmp/test_save.agent'
    hook = AgentHook(agent, save_path=save_path)

    retrieved_agent = AgentHook.unhook(hook)
    model_list = [retrieved_agent.algorithm.model, retrieved_agent.algorithm.target_model]
    assert_model_parameters_are_cuda_tensors(model_list)


def test_can_load_ppo_from_agenthook_with_cuda(RPSTask, ppo_config_dict):
    ppo_config_dict['use_cuda'] = True
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    save_path = '/tmp/test_save.agent'
    hook = AgentHook(agent, save_path=save_path)

    assert not hasattr(hook, 'agent')

    retrieved_agent = AgentHook.unhook(hook)
    model_list = [retrieved_agent.algorithm.model]
    assert_model_parameters_are_cuda_tensors(model_list)


def test_can_load_ppo_from_agenthook_disabling_cuda(RPSTask, ppo_config_dict):
    ppo_config_dict['use_cuda'] = True
    agent = build_PPO_Agent(RPSTask, ppo_config_dict, 'PPO')
    save_path = '/tmp/test_save.agent'
    hook = AgentHook(agent, save_path=save_path)

    retrieved_agent = AgentHook.unhook(hook, use_cuda=False)
    model = retrieved_agent.algorithm.model
    assert all(map(lambda param: not param.is_cuda, model.parameters()))


def assert_model_parameters_are_cuda_tensors(model_list):
    for model in model_list: assert all(map(lambda param: param.is_cuda, model.parameters()))


def assess_file_has_been_saved_on_disk_and_not_on_ram(hook, save_path):
    assert not hasattr(hook, 'agent')
    assert hook.save_path is save_path
    assert os.path.exists(save_path)
