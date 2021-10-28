from typing import List, Union, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_winrate_matrix(winrate_matrix: Union[List, np.ndarray],
                        show_annotations=True, cbar=False,
                        vmin: float=0.,
                        vmax: float=1.,
                        xlabels: Optional[List[str]]=None,
                        ylabels: Optional[List[str]]=None,
                        ax: Optional[plt.Axes] = None) \
                        -> plt.Axes:
    '''
    Plots the :param: winrate matrix: on a heatmap.

    Red values mean < 50% winrates
    Positive values are shown in blue.
    If :param: ax is not present
    We'll create one for you <3

    :param winrate_matrix: Winrate matrix to plot. Values must be within [0, 1]
    :param vmin: Minimum value for the matrice's colourscheme
    :param vmax: Maximum value for the matrice's colourscheme
    :param xlabels: Labels for each of the xticks in the output plot
    :param ylabels: Labels for each of the yticks in the output plot
    :param ax: Ax where the plot should be plotted. Optional
    :show annotations: Flag determining whether values inside of the heatmap should be written
    :returns: ax where the winrate_matrix has been plotted
    '''
    if not ax:
        ax = plt.subplot(111)

    heatmap = sns.heatmap(winrate_matrix, annot=show_annotations, ax=ax, square=True,
                cmap=sns.color_palette('coolwarm', 50)[::-1],
                vmin=vmin, vmax=vmax,
                cbar=cbar, cbar_kws={'label': 'Head to head winrates'},
    )
    heatmap.set_yticklabels(heatmap.get_yticklabels(),
                            rotation=0)
    if xlabels: heatmap.set_xticklabels(xlabels)
    if ylabels: heatmap.set_yticklabels(ylabels)
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Agent ID')
    # This does not seem needed any more. Leaving it here because
    # it might be required for streamlit
    #ax.set_ylim(len(winrate_matrix) + 0.2, -0.2)  # Required seaborn hack
    ax.set_title('Empirical winrate matrix')
    return ax
