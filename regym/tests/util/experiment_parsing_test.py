import yaml
import pytest

from regym.util.experiment_parsing import filter_relevant_configurations


@pytest.fixture
def experiment_config():
    experiment_config = '''
    experiment:
        algorithms: ['ppo', 'tabularqlearning']
        self_play_training_schemes: ['psro']
    agents:
        deepqlearning:
            placeholder: None
        ppo:
            placeholder: None
        ppo_dani:
            placeholder: None
    self_play_training_schemes:
        psro:
            placeholder: None
        delta_distributional:
            placeholder: None
    '''
    return yaml.load(experiment_config)


def test_can_filter_agent_configuration_based_on_experiment_algorithms(experiment_config):
    filtered_agent_config = filter_relevant_configurations(experiment_config['experiment'],
                                                           target_configs=experiment_config['agents'],
                                                           target_key='algorithms')
    assert 'ppo' in filtered_agent_config
    assert 'ppo_dani' in filtered_agent_config
    assert 'deeplearning' not in filtered_agent_config
    assert 'tabularqlearning' not in filtered_agent_config

    filtered_sp_config = filter_relevant_configurations(experiment_config['experiment'],
                                                        target_configs=experiment_config['self_play_training_schemes'],
                                                        target_key='self_play_training_schemes')
    assert 'psro' in filtered_sp_config
    assert 'delta_distributional' not in filtered_sp_config
