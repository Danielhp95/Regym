from typing import Dict, List

import torch

class Storage:
    def __init__(self, size, keys: List[str] = None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'non_terminal',
                       'V', 'Q', 'pi', 'log_pi', 'entropy',
                       'adv', 'ret', 'q_a', 'log_pi_a', 'probs',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add_key(self, key: str):
        self.keys += [key]
        setattr(self, key, [])

    def add(self, data: Dict):
        for k, v in data.items():
            assert k in self.keys
            storage_element = getattr(self, k)

            # Hack: if it's list of lists
            if isinstance(v, list) and isinstance(v[0], list):
                storage_element += v
            elif isinstance(v, list):
                storage_element += v
            else: storage_element.append(v)

    def placeholder(self, num_elements: int = -1):
        if num_elements == -1:
            num_elements = self.size
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * num_elements)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def non_empty_keys(self):
        return [k for k in self.keys if getattr(self, k) != []]

    def cat(self, keys: List[str]):
        data = [getattr(self, k)[:self.size] for k in keys]
        return data

    def get(self, key: str):
        return getattr(self, key)

    def remove(self, i: int):
        '''Removes :param: index i from all keys that feature it'''
        for k in self.keys:
            if i < len(self.get(k)): self.get(k).pop(i)

    def remove_duplicates(self, target_key: str, avg_keys: List[str] = []):
        '''
        Removes duplicate of :param: target_key.
        If keys are present in :param: avg_keys, the average of the values
        in those keys are used to replace the remaining copy's value.

        :return: Number of duplicate entries
        '''
        # ASSUMPTION: all populated keys have the same number of items
        total_duplicates = 0
        size = len(self.get(target_key))
        for i in range(size):
            if i >= len(self.get(target_key)): break
            x = self.get(target_key)[i]
            duplicate_indices = self.find_duplicates(x, target_key, i)
            if len(duplicate_indices) > 0:
                total_duplicates += len(duplicate_indices)
                #print(f'{len(duplicate_indices)} duplicates found at index {i}')
                if avg_keys != []:
                    k_avg = self.compute_average_over_keys([i] + duplicate_indices, avg_keys)
                    for k in avg_keys: self.get(k)[i] = k_avg[k]
                for j in sorted(duplicate_indices, reverse=True): self.remove(j)
        #print(size, total_duplicates, total_duplicates / size)


    def find_duplicates(self, x, target_key, start_index: int = -1):
        return [j for j in range(start_index + 1, len(self.get(target_key)))
                if torch.equal(x, self.get(target_key)[j])]

    def compute_average_over_keys(self, duplicate_indices, avg_keys) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([self.get(k)[i]
                                for i in duplicate_indices]).mean(dim=0)
                for k in avg_keys}

    def __repr__(self):
        keys_and_item_numbers = [('key: ' + k, 'items: ' + str(len(getattr(self, k))))
                                 for k in self.non_empty_keys()]
        return f'Storage. Size: {self.size}. Used keys: {keys_and_item_numbers}'
