import numpy as np
from collections import namedtuple

EXPPER = namedtuple('EXPPER', ('idx','priority','state','action','next_state', 'reward','done') )


class PrioritizedReplayBuffer :
    def __init__(self,capacity, alpha=0.2) :
        self.length = 0
        self.counter = 0
        self.alpha = alpha
        self.epsilon = 1e-6
        self.capacity = int(capacity)
        self.tree = np.zeros(2*self.capacity-1)
        self.data = np.zeros(self.capacity,dtype=object)
        self.sumPi_alpha = 0.0

    def save(self,path):
        path += '.prb'
        np.savez(path, tree=self.tree, data=self.data,
            length=np.asarray(self.length), sumPi=np.asarray(self.sumPi_alpha),
            counter=np.asarray(self.counter), alpha=np.asarray(self.alpha) )

    def load(self,path):
        path += '.prb.npz'
        data= np.load(path)
        self.tree =data['tree']
        self.data = data['data']
        self.counter = int(data['counter'])
        self.length = int(data['length'])
        self.sumPi_alpha = float(data['sumPi'])
        self.alpha = float(data['alpha'])

    def reset(self) :
        self.__init__(capacity=self.capacity,alpha=self.alpha)

    def add(self, exp, priority) :
        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity

        idx = self.counter + self.capacity -1

        self.data[self.counter] = exp

        self.counter += 1
        self.length = min(self.length+1, self.capacity)
        if self.counter >= self.capacity :
            self.counter = 0

        self.sumPi_alpha += priority
        self.update(idx,priority)

    def priority(self, error) :
        return (error+self.epsilon)**self.alpha

    def update(self, idx, priority) :
        if np.isnan(priority) or np.isinf(priority) :
            priority = self.total()/self.capacity

        change = priority - self.tree[idx]
        if change > 1e3 :
            print('BIG CHANGE HERE !!!!')
            print(change)
            raise Exception()

        previous_priority = self.tree[idx]
        self.sumPi_alpha -= previous_priority

        self.sumPi_alpha += priority
        self.tree[idx] = priority

        self._propagate(idx,change)

    def _propagate(self, idx, change) :
        parentidx = (idx - 1) // 2

        self.tree[parentidx] += change

        if parentidx != 0 :
            self._propagate(parentidx, change)

    def __call__(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1
        data = self.data[dataidx]
        priority = self.tree[idx]

        return (idx, priority, data)

    def get(self, s) :
        idx = self._retrieve(0,s)
        dataidx = idx-self.capacity+1

        data = self.data[dataidx]
        if not isinstance(data,EXP) :
            raise TypeError

        priority = self.tree[idx]

        return (idx, priority, *data)

    def get_importance_sampling_weight(priority,beta=1.0) :
        return pow( self.capacity * priority , -beta )

    def get_buffer(self) :
        return [ self.data[i] for i in range(self.capacity) if isinstance(self.data[i],EXP) ]


    def _retrieve(self,idx,s) :
         leftidx = 2*idx+1
         rightidx = leftidx+1

         if leftidx >= len(self.tree) :
            return idx

         if s <= self.tree[leftidx] :
            return self._retrieve(leftidx, s)
         else :
            return self._retrieve(rightidx, s-self.tree[leftidx])

    def total(self) :
        return self.tree[0]

    def __len__(self) :
        return self.length
