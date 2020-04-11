from typing import Dict, List


class Storage:
    def __init__(self, size, keys: List[str] = None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'non_terminal',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add_key(self, key: str):
        self.keys += [key]
        setattr( self, key, [])
        
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

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys: List[str]):
        data = [getattr(self, k)[:self.size] for k in keys]
        return data

    def __repr__(self):
        keys_and_item_numbers = [('key: ' + k, 'items: ' + str(len(getattr(self, k))))
                                 for k in self.keys if getattr(self, k) != []]
        return f'Storage. Size: {self.size}. Used keys: {keys_and_item_numbers}'
