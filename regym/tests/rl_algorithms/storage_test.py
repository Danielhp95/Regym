from regym.rl_algorithms.replay_buffers import Storage

import torch

def generate_storage():
    s = Storage(size=10, keys=['k1', 'k2'])
    s.add({'k1': torch.Tensor([1]), 'k2': torch.Tensor([0, 0])})
    s.add({'k1': torch.Tensor([1]), 'k2': torch.Tensor([1, 1])})
    s.add({'k1': torch.Tensor([1]), 'k2': torch.Tensor([2, 2])})
    # One of this should be removed, average is torch.tensor([2, 2])
    s.add({'k1': torch.Tensor([2]), 'k2': torch.Tensor([1, 1])})
    s.add({'k1': torch.Tensor([2]), 'k2': torch.Tensor([3, 3])})
    s.add({'k1': torch.Tensor([3]), 'k2': torch.Tensor([0, 0])})
    return s


def test_can_remove_duplicates():
    expected_size = 3
    storage = generate_storage()

    storage.remove_duplicates(target_key='k1')

    assert len(storage.k1) == expected_size
    assert len(storage.k2) == expected_size


def test_can_remove_average_over_duplicate_values():
    expected_size = 3
    storage = generate_storage()

    storage.remove_duplicates(target_key='k1', avg_keys=['k2'])

    assert len(storage.k1) == expected_size
    assert len(storage.k2) == expected_size
    assert torch.equal(storage.k2[0], torch.Tensor([1, 1]))
    assert torch.equal(storage.k2[1], torch.Tensor([2, 2]))
    assert torch.equal(storage.k2[2], torch.Tensor([0, 0]))
