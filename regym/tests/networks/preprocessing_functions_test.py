import torch
import numpy as np

from regym.networks.preprocessing import (batch_vector_observation,
                                          flatten_and_turn_into_batch,
                                          turn_into_single_element_batch,
                                          flatten_last_dim_and_batch_vector_observation)


#def test_turn_into_single_element_batch():
#    assert False


#def test_flatten_and_turn_into_batch():
#    assert False

#def test_batch_vector_observation_from_list():
#    assert False


def test_batch_vector_observation_from_numpy_vectors():
    input_array_1 = np.array([1,2,3])
    input_array_2 = np.array([4,5,6])
    expected_tensor_1 = torch.FloatTensor([[1,2,3]])
    expected_tensor_2 = torch.FloatTensor([[1,2,3],
                                           [4,5,6]])

    actual_tensor_1 = batch_vector_observation([input_array_1])

    assert torch.equal(actual_tensor_1, expected_tensor_1)

    actual_tensor_2 = batch_vector_observation([input_array_1, input_array_2])

    assert torch.equal(actual_tensor_2, expected_tensor_2)


def test_batch_vector_observation_from_numpy_matrices():
    input_array_1 = np.array([[1,2,3],
                              [4,5,6]])
    input_array_2 = np.array([[5,6,7],
                              [8,9,10]])

    expected_tensor_1 = torch.FloatTensor([[[1,2,3], [4,5,6]]])
    expected_tensor_2 = torch.FloatTensor([[[1,2,3], [4,5,6]],
                                           [[5,6,7], [8,9,10]]])

    actual_tensor_1 = batch_vector_observation([input_array_1])

    assert torch.equal(actual_tensor_1, expected_tensor_1)

    actual_tensor_2 = batch_vector_observation([input_array_1, input_array_2])

    assert torch.equal(actual_tensor_2, expected_tensor_2)


def test_flatten_last_dim_and_batch_vector_observation():
    # Shape: 3 x 1 x 2
    partial_input_array_1 = np.array([[[0,0]],
                                      [[1,1]],
                                      [[2,2]]])
    # Shape: 3 x 1 x 2
    partial_input_array_2 = np.array([[[3,3]],
                                      [[4,4]],
                                      [[5,5]]])

    # Shape: 2 x 3 x 1 x 2
    # Note: it's a list, so it doesn't have attribute shape, but it OK
    input_array = [partial_input_array_1,
                   partial_input_array_2]

    # Shape: 2 x 3 x 2
    expected_tensor = torch.FloatTensor([[[0,0],
                                          [1,1],
                                          [2,2]],
                                         [[3,3],
                                          [4,4],
                                          [5,5]]])

    actual_tensor = flatten_last_dim_and_batch_vector_observation(input_array)

    assert torch.equal(actual_tensor, expected_tensor)
