import os
import sys
sys.path.append(os.path.abspath('../../'))

import yaml
import pytest

from util import experiment_parsing


@pytest.fixture
def experiment_config():
    experiment_config = '''
    experiment:
        algorithms: ['ppo', 'tabularqlearning']
    agents:
        deepqlearning:
            placeholder: None
        ppo:
            placeholder: None
        ppo_dani:
            placeholder: None
    '''
    return yaml.load(experiment_config)


def test_can_filter_agent_configuration_based_on_experiment_algorithms(experiment_config):
    filtered_agent_config = experiment_parsing.filter_relevant_agent_configurations(experiment_config['experiment'], experiment_config['agents'])
    assert 'ppo' in filtered_agent_config
    assert 'ppo_dani' in filtered_agent_config
    assert 'deeplearning' not in filtered_agent_config
    assert 'tabularqlearning' not in filtered_agent_config
