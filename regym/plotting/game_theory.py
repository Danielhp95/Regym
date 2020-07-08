from typing import List, Union, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_winrate_matrix(winrate_matrix: Union[List, np.ndarray],
                        ax: Optional[plt.Axes] = None, show_annotations=True) \
                        -> plt.Axes:
    '''
    Plots the :param: winrate matrix: on a heatmap.

    Red values mean < 50% winrates
    Positive values are shown in blue.
    If :param: ax is not present
    We'll create one for you <3

    :param winrate_matrix: Winrate matrix to plot. Values must be within [0, 1]
    :param ax: Ax where the plot should be plotted. Optional
    :show annotations: Flag determining whether values inside of the heatmap should be written
    :returns: ax where the winrate_matrix has been plotted
    '''
    if not ax:
        ax = plt.subplot(111)

    sns.heatmap(winrate_matrix, annot=show_annotations, ax=ax, square=True,
                cmap=sns.color_palette('coolwarm', 50)[::-1],
                vmin=0.0, vmax=1.0, cbar=False, cbar_kws={'label': 'Head to head winrates'})
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Agent ID')
    ax.set_ylim(len(winrate_matrix) + 0.2, -0.2)  # Required seaborn hack
    ax.set_title('Empirical winrate matrix')
    return ax
