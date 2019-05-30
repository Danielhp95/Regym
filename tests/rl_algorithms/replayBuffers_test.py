from regym.rl_algorithms.replay_buffers import PrioritizedReplayBuffer, EXP
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def test_prioritizedReplayBuffer_instantiation():
    '''
    Create histograms of samples per items in the replay buffer.
    Each histogram is computed with a different value of alpha
    (used to compute the priority from the error).
    From 0.1 to 1.0 (biasing towards high-error items).
    '''
    num_bins = 1000
    nbr_iterative_sampling = 1000

    capacity = 1000
    alphas = [1.0, 0.8, 0.6, 0.3, 0.1]
    colors = ['yellow', 'green', 'blue', 'red', 'grey']

    batch_size = 52
    fraction = 0.0

    for alpha, color in zip(alphas, colors):
        replayBuffer = PrioritizedReplayBuffer(capacity=capacity, alpha=alpha)
        experience = EXP(None, None, None, None, None)
        for i in range(capacity):
            importance = (i/capacity)*np.ones((1, 1))
            init_sampling_priority = replayBuffer.priority(importance)
            replayBuffer.add(experience, init_sampling_priority)

        prioritysum = replayBuffer.total()

        batch = list()
        importances = list()

        for i in range(nbr_iterative_sampling):
            # Random Experience Sampling with priority
            low = fraction*prioritysum
            step = (prioritysum-low) / batch_size
            randexp = np.arange(low, prioritysum, step)+np.random.uniform(low=0.0, high=step, size=(batch_size))

            for i in range(batch_size):
                try:
                    el = replayBuffer.get(randexp[i])
                    importances.append(np.exp(np.log(el[1])/alpha))
                    batch.append(el)
                except TypeError:
                    continue

        n, bins, patches = plt.hist(importances, num_bins, facecolor=color, alpha=0.3)

    plt.xlabel("Item's importance in the replay buffer")
    plt.ylabel("Sampling count of items with given importance")
    plt.title(r"Average histogram of sampling count over prioritized replay buffer across various $\alpha$")

    handles = [mpatches.Patch(color=color, label=r'$\alpha$ = {}'.format(alpha))
               for alpha, color in zip(alphas, colors)]
    plt.legend(handles=handles, fontsize='x-large')

    plt.show()


if __name__ == "__main__":
    test_prioritizedReplayBuffer_instantiation()
