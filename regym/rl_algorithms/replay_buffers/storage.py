class Storage:
    def __init__(self, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'succ_s', 'non_terminal',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean', 'action_logits']
        self.keys = keys
        self.reset()

    def add_key(self, key):
        self.keys += [key]
        setattr(self, key, [])

    def add(self, data):
        for k, v in data.items():
            assert k in self.keys, f'Tried to add value key ({k}, {v}) but, {k} is not a registered key'
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * (len(self)-1))

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k) for k in keys]
        return data

    def __len__(self):
        lengths = []
        for k in self.keys:
            lengths.append(len(getattr(self,k)))
        max_length = max(lengths)
        return max_length 

    def __repr__(self):
        string_form = 'Storage:\n'
        for k in self.keys:
            v = getattr(self, k)
            if v != []:
                string_form += f'{k}: {v}\n'
        return string_form
