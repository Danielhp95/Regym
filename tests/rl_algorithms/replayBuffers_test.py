import os
import sys
sys.path.append(os.path.abspath('../../'))

from rl_algorithms.replay_buffers import PrioritizedReplayBuffer, EXP
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


def test_prioritizedReplayBuffer_instantiation():
    '''
    Create histograms of samples per items in the replay buffer.
    Each histogram is computed with a different value of alpha 
    (used to compute the priority from the error).
    From 0.00001 (uniform distribution) to 1.0 (biasing towards high-error items).
    '''
    num_bins = 1000
    nbr_iterative_sampling = 1000

    capacity = 1000
    alphas = [1.0,0.8,0.6,0.3,0.1,0.00001]
    colors = ['yellow','green','blue','red','grey','black']

    batch_size = 256
    fraction = 0.0#0.5#0.9
    

    

    for alpha,color in zip(alphas,colors):
        replayBuffer = PrioritizedReplayBuffer(capacity=capacity,alpha=alpha)
        experience = EXP(None,None,None,None,None)
        for i in range(capacity):
            importance = (i/capacity)*np.ones((1,1) )
            #importance = (i)*np.ones((1,1) )
            init_sampling_priority = replayBuffer.priority( importance )
            replayBuffer.add(experience, init_sampling_priority)

        prioritysum = replayBuffer.total()

        batch = list()
        priorities = list()
        importances = list()

        for i in range(nbr_iterative_sampling):
            # Random Experience Sampling with priority
            low = fraction*prioritysum
            step = (prioritysum-low) / batch_size
            try:
                randexp = np.arange(low,prioritysum,step)+np.random.uniform(low=0.0,high=step,size=(batch_size))
            except Exception as e :
                print( prioritysum, step)
                raise e
            
            for i in range(batch_size):
                try :
                    el = replayBuffer.get(randexp[i])
                    priorities.append( el[1] )
                    importances.append( np.exp(np.log(el[1])/alpha) )
                    batch.append(el)
                except TypeError as e :
                    continue

        #n, bins, patches = plt.hist(priorities, num_bins, facecolor=color, alpha=0.8)
        n, bins, patches = plt.hist(importances, num_bins, facecolor=color, alpha=0.8)
        
    plt.show()

if __name__ == "__main__" :
    test_prioritizedReplayBuffer_instantiation()