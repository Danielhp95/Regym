from typing import Tuple, Iterator

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

from torchviz import make_dot


def plot_gradient_flow(named_parameters: Iterator[Tuple[(str, Parameter)]]):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage:

    >>> model: nn.Module
    >>> plot_gradient_flow(model.named_parameters())

    to visualize the gradient flow
    Taken from : https://github.com/alwynmathew/gradflow-check
    '''

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


def plot_backwards_graph(tensor: Tensor, model: torch.nn.Module, filename: str):
    '''
    Generates a plot containng :param: tensor's backward computational graph,
    which will be stored in file :param: {filename}.pdf.
    NOTE: - Two files are created (1) filename (2) filename.pdf
          - This function WILL NOT plot in a terminal, only to a file

    :param tensor: Tensor's backward computational path
    :param model: Model from which the tensor originated
    :param filename: Name of the pdf file where the plot will be stored
    '''
    graph = make_dot(tensor, params=dict(model.named_parameters()))
    graph.render(filename=filename, format='pdf')