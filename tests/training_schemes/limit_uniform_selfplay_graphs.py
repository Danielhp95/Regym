import numpy
from numpy.random import choice
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def uniform_sampling(max_size, samples_before_addition):
    menagerie = {}
    for i in range(max_size):
        menagerie[i] = 0
        keys = menagerie.keys()
        print('Uniform Iteration: {}/{}'.format(i, max_size), end='\r')
        probabilities = [1/len(keys) for _ in range(len(keys))]
        for _ in range(samples_before_addition):
            random_policy = choice(list(keys), p=probabilities)
            menagerie[random_policy] += 1

    number_of_samples = max_size * samples_before_addition

    keys = menagerie.keys()
    values = menagerie.values()
    sample_rates = [v/number_of_samples for v in values]
    plt.bar(keys, sample_rates, color='r', alpha=0.1)


def maybe_uniform(max_size, samples_before_addition):
    menagerie = {}
    for i in range(max_size):
        menagerie[i] = 0
        keys = menagerie.keys()
        n = len(menagerie)
        unormalized_ps = [samples_before_addition/((n * (n-i)**2)) for i in range(n)]
        normalize_ps = [p / sum(unormalized_ps) for p in unormalized_ps]
        print('Limit uniform Iteration: {}/{}'.format(i, max_size), end='\r')
        for _ in range(samples_before_addition):
            random_policy = choice(list(keys), p=normalize_ps)
            menagerie[random_policy] += 1

    number_of_samples = max_size * samples_before_addition

    keys = menagerie.keys()
    values = menagerie.values()

    sample_rates = [v/number_of_samples for v in values]

    plt.bar(keys, sample_rates, color='b', alpha=0.5)


if __name__ == '__main__':
    # Maximum number of policies that will be added 
    max_size = 500
    # Number of times that policies will be sampled before a new one is added to the menagerie
    samples_before_addition = 2

    plt.ylabel('Sample rate', fontsize=20)
    plt.xlabel('Policy indexes', fontsize=20)

    plt.tick_params(labelsize=15)

    # Creating legend
    red_patch = mpatches.Patch(color='red', label=r'$\delta$-uniform. $\delta=0$')
    blue_patch = mpatches.Patch(color='blue', label='Limit uniform')
    plt.legend(handles=[red_patch, blue_patch], fontsize='x-large')

    # Plotting uniform distribution between 0 and maxsize with an orange line
    uniform_expectation_line = [[-1, max_size], [1/max_size, 1/max_size]]
    plt.plot(uniform_expectation_line[0], uniform_expectation_line[1], '-')

    # Calculate and sample empirical sample rate of delta-uniform self-play
    maybe_uniform(max_size=max_size, samples_before_addition=samples_before_addition)
    # Calculate and sample empirical sample rate of Limit uniform policy sampling distribution
    uniform_sampling(max_size=max_size, samples_before_addition=samples_before_addition)

    plt.show()
    plt.savefig('sample-rates.eps', format='eps', dpi=1000)
