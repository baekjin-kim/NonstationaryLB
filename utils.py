#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    File description: Useful functions
"""

# Modified Author: Baekjin Kim (baekjin@umich.edu)
# Copyright (c) 2019, Yoan Russac (yoan.russac@ens.fr)
# License: BSD (3-clause)

# Importations
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

current_palette = sns.color_palette()
sns.set_style("ticks")
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc("lines", linewidth=3)
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
matplotlib.rc('font', weight='bold')
matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath} \boldmath"]
styles = ['o', '^', 's', 'D', 'p', 'v', '*','o', '^', 's', 'D', 'p', 'v', '*', '^', 's', 'D', 'p', 'v', '*']
colors = current_palette[0:11]


def plot_regret(data, t_saved, filename, log=False, qtl=False, loc=0, font=10, bp=None, bp_2=None):
    """
    param:
        - data:
        - t_saved: numpy array (ndim = 1), index of the points to save on each trajectory
        - filename: Name of the file to save the plot of the experiment, if None then it is only plotted
        - log: Do you want a log x-scale
        - qtl: Plotting the lower and upper quantiles. Other effect: If qtl == False then only t_saved
               are printed in the other case everything is printed
        - loc: Location of the legend for fine-tuning the plot
        - font: Font of the legend for fine-tuning the plot
        - bp: Dictionary for plotting the time steps where the breakpoints occur
        - bp_2: Dictionary for plotting the time steps where the breakpoints where detected for d-LinUCB
    Output:
    -------
    Plot it the out/filename file
    """
    linestyle = ['solid', 'solid','dashed', 'solid', 'solid','dashed','solid', 'solid','dashed']
    fig = plt.figure(figsize=(7, 6))
    if log:
        plt.xscale('log')
    i = 0

    if t_saved is None:
        len_tsaved = len(data[0][1])
        t_saved = [i for i in range(0,len_tsaved)]

    for key, avgRegret, qRegret, QRegret in data:
        label = r"\textbf{%s}" % key
        plt.plot(t_saved, avgRegret, marker=styles[i],
                 markevery=0.1, ms=10.0, label=label, color=colors[i], linestyle=linestyle[i])
        if qtl:
            plt.fill_between(t_saved, qRegret, QRegret, alpha=0.15,
                             linewidth=1.5, color=colors[i])
        i += 1
    plt.legend(loc=loc, fontsize=font).draw_frame(True)
    plt.xlabel(r'Round $\boldsymbol{t}$', fontsize=20)
    plt.ylabel(r'Regret $\boldsymbol{R(T)}$', fontsize=18)
    for x in bp:
        plt.axvline(x, color='red', linestyle='--', lw=1)
    for x in bp_2:
        plt.axvline(x, color='blue', linestyle='--', lw=1)
    if filename:
        plt.savefig('%s.png' % filename, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return
