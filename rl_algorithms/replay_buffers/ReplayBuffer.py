import numpy as np
from .experience import EXP, EXPPER

class ReplayBuffer(object) :
    def __init__(self,capacity) :
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) :
        if len(self.memory) < self.capacity :
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1) % self.capacity
        self.position = int(self.position)

    def sample(self,batch_size) :
        return random.sample(self.memory, batch_size)

    def __len__(self) :
        return len(self.memory)

    def save(self,path):
        path += '.rb'
        np.savez(path, memory=self.memory, position=np.asarray(self.position) )

    def load(self,path):
        path += '.rb.npz'
        data= np.load(path)
        self.memory =data['memory']
        self.position = int(data['position'])
