import pytest
import numpy as np

from regym.util.data_augmentation import generate_horizontal_symmetry


@pytest.fixture()
def connect4_random_exp():
    random_observation = np.array(
        [[[0, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1]],
        [[1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]]
    )
    a = 1
    r = 0.
    succ_o = random_observation
    done = False
    extra_info = {'self': {'probs':
                           np.array([1, 2, 3, 4, 5, 6, 7])},
                  0:    {'probs':
                           np.array([1, 2, 3, 4, 5, 6, 7])}
                 }
    return random_observation, a, r, succ_o, done, extra_info


@pytest.fixture()
def connect4_symmetric_random_exp():
    random_observation = np.array(
        [[[1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1]],
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]]]
    )
    a = 1
    r = 0.
    done = False
    succ_o = random_observation
    extra_info = {'self': {'probs':
                           np.array([7, 6, 5, 4, 3, 2, 1])},
                  0:    {'probs':
                           np.array([7, 6, 5, 4, 3, 2, 1])}
                  }
    return random_observation, a, r, succ_o, done, extra_info


def test_connect4_symmetric_data_augmentation(connect4_random_exp, connect4_symmetric_random_exp):
    o, a, r, succ_o, done, extra_info = connect4_random_exp
    target_o, target_a, target_r, target_succ_o, target_done, target_extra_info = connect4_symmetric_random_exp
    actual_o, actual_a, actual_r, actual_succ_o, actual_done, actual_extra_info =  generate_horizontal_symmetry(o, a, r, succ_o, done, extra_info, flip_obs_on_dim=1)[0]

    np.testing.assert_array_equal(actual_o, target_o)
    assert actual_r == target_r
    assert actual_a == target_a
    assert actual_done == target_done
    np.testing.assert_array_equal(actual_extra_info['self']['probs'], target_extra_info['self']['probs'])
    np.testing.assert_array_equal(actual_extra_info[0]['probs'], target_extra_info[0]['probs'])
    # TODO: potentially also change observation inside of target extra info
    # Do something about succ_o?
