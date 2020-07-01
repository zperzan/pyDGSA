# -*- coding: utf-8 -*-

"""
pyDGSA sub-package that provides useful utilities for plotting dgsa results

This file includes:
vert_pareto_plot
plot_cdf
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from matplotlib.colors import to_rgba

def vert_pareto_plot(df, np_plot='+5', fmt=None, colors=None):
    """Generate a vertical Pareto plot of sensitivity analysis results.
    
    params:
        df [DataFrame]: pandas dataframe containing the sensitivity analysis
                results. If fmt == 'max' or 'mean', df contains a single column
                with indices correspond to the parameters. If fmt == 'cluster_avg', 
                the columns of df correspond to the number of clusters and 
                the rows correspond to the parameters.
        np_plot [str|int]: number of parameters to plot. Default: '+5'
                -'all': plot all parameters
                -n: plot n parameters, where n is an int
                -'+n': plot all parameters with sensitivity >1, plus 
                       the n next most sensitive parameters
        fmt [str]: format of df. Optional, will interpret fmt based on df shape
                -'mean': single sensitivity was passed per param, therefore
                         display a single bar per param
                -'max': same as 'mean'
                -'cluster_avg': cluster-specific sensitivities were passed, so
                            display each cluster's sensitivity separately
                -'bin_avg': bin-specific sensitivities were passed, so display
                            each bin's sensitivity separately
                -'indiv': plots the sensitivity for each bin/cluster combination
                        separately
        colors [list(str|int)]: list of clusters colors to use when plotting, 
                either specified as rgba tuples or strings of matplotlib named 
                colors. Only used when fmt='cluster_avg' or 'indiv'
    
    returns:
        fig: matplotlib figure handle
        ax: matplotlib axis handle
    """
    
    # Total np (number of parameters)
    np_total = df.shape[0] 
    
    # Figure out fmt if not explicitly provided
    if fmt is None:
        if isinstance(df.columns, pd.MultiIndex):
            fmt = 'indiv'
        elif df.shape[1] > 1:
            # Could be either 'cluster_avg' or 'bin_avg'
            if 'Cluster' in df.columns[0]:
                fmt = 'cluster_avg'
            elif 'Bin' in df.columns[0]:
                fmt = 'bin_avg'
            else:
                raise ValueError("Could not determine fmt. Please pass explicitly.")
        else:
            fmt = 'mean'
    
    if fmt == 'indiv' and colors is not None:
        if isinstance(colors[0], str):
            # Convert named colors to rgba
            named_colors = colors
            colors=[]
            for color in named_colors:
                colors.append(to_rgba(color))
    
    # Get number of parameters with sensitivity >= 1
    np_sensitive = np.any((df.fillna(0).values >= 1), axis=1).sum()

    # Figure out how many parameters to plot
    if isinstance(np_plot, str):
        if np_plot == 'all':
            np_max_plot = np_total
        elif np_plot[0] == '+':
            np_max_plot = np_sensitive + int(np_plot[1:])
        else:
            raise ValueError("np_plot must be 'all', 'n', or '+n', where n is an int")
    elif isinstance(np_plot, int):
        np_max_plot = np_plot

    # Ensure that requested # of params to plot is not larger than total # of params
    if np_max_plot > np_total:
        np_max_plot = np_total

    # Y-position of bars
    y_pos = np.arange(np_max_plot)
    
    if fmt == 'mean' or fmt == 'max':
        # Sort so most sensitive params are on top
        df.sort_values(by=df.columns[0], ascending=False, inplace=True)
        data = df.values[:np_max_plot, :].squeeze()
        params = df.index.tolist() # Get list of params after sorting
        
        yticks = y_pos 

        # Initialize colors as red unless there are no sensitive parameters,
        # in which case initialize as blue
        if np_sensitive > 0:
            colors = np.asarray([[1, 0, 0, 0.8]]*np_max_plot)
        else:
            colors = np.asarray([[0, 0, 1, 0.8]]*np_max_plot)

        if np_max_plot > np_sensitive:
            if np_sensitive > 0:
                colors[np_sensitive] = [1, 1, 1, 1]
            colors[np_sensitive+1:] = [0, 0, 1, 0.8] 

        fig_height = int(np_max_plot/2)
        
        # Create figure and add barh
        fig, ax = plt.subplots(figsize=(5, fig_height))
        ax.barh(y_pos, data, color=colors, edgecolor='k')

    elif fmt == 'cluster_avg':
        n_clusters = df.shape[1]
        
        # Sort by mean sensitivity across clusters 
        sort_df = df.mean(axis=1).sort_values(ascending=False)
        df = df.reindex(sort_df.index)
        params = df.index.tolist() # Get list of params after sorting
        
        height = 1/(n_clusters+1)
        yticks = y_pos - (height*(n_clusters-1)/2)
        
        if colors is None:
            colors = []
            cmap = matplotlib.cm.get_cmap('Set1')
            for i in range(n_clusters):
                colors.append(cmap(i))

        # Create figure
        fig_height = int(np_max_plot/2*1.5)
        fig, ax = plt.subplots(figsize=(5, fig_height))

        # Add bars for each cluster
        for i in range(n_clusters):
            ax.barh(y_pos - height*i, df.iloc[:np_max_plot, i], height=height, 
                    color=colors[i], edgecolor='k', label=df.columns.tolist()[i])
        ax.legend()
        
    elif fmt == 'bin_avg':
        n_bins = df.shape[1]
        
        # Sort by mean sensitivity across bins 
        sort_df = df.mean(axis=1).sort_values(ascending=False)
        df = df.reindex(sort_df.index)
        params = df.index.tolist()

        yticks = y_pos

        if colors is None:
            cmap = matplotlib.cm.get_cmap('Set1')
            colors = cmap(1)

        # Create color array by decreasing alpha channel for each bin
        color_array = np.tile(colors, (n_bins, 1))
        for i in range(n_bins):
            color_array[i, 3] = (i+1)/(n_bins)

        fig_height = int(np_max_plot/2*1.5)
        fig, ax = plt.subplots(figsize=(5, fig_height))

        for i in range(n_bins):
            width = df.iloc[:np_max_plot, i]
            left = df.iloc[:np_max_plot, :i].sum(axis=1)
            b = ax.barh(y_pos, width=width, left=left, color=color_array[i], 
                        edgecolor='k')
                
            # Increase linewidth for parameters that are sensitive
            for w in enumerate(width.tolist()):
                if w[1] > 1:
                    b[w[0]].set_linewidth(2.5)
        
    elif fmt == 'indiv':
        n_clusters, n_bins = df.columns.levshape
        
        # Split df into sensitive and non-sensitive df's, sort each, then re-combine
        # Can't just sort on mean sensitivity, otherwise a sensitive parameter
        # could get left out because its mean might not be within the top 
        # np_max_plot most sensitive, even though a single bin is >= 1
        mask = np.any((df.fillna(0).values >= 1), axis=1)
        sens = df[mask].copy()
        nsens = df[~mask].copy()
        sort_sens = sens.mean(axis=1).sort_values(ascending=False)
        sens = sens.reindex(sort_sens.index)
        sort_nsens = nsens.mean(axis=1).sort_values(ascending=False)
        nsens = nsens.reindex(sort_nsens.index)
        df = sens.append(nsens)

        params = df.index.tolist()
        df.fillna(0, inplace=True)
        height = 1/(n_clusters+1)
        yticks = y_pos - (height*(n_clusters-1)/2)

        if colors is None:
            cmap = matplotlib.cm.get_cmap('Set1')
            colors = []
            for i in range(n_clusters):
                colors.append(cmap(i))

        idx = pd.IndexSlice
                
        # Create color array by decreasing alpha channel for each bin
        color_array = np.zeros((n_clusters, n_bins, 4), dtype='float64')
        for i in range(n_clusters):
            for j in range(n_bins):
                color_array[i, j, :] = colors[i]
                color_array[i, j, 3] = (j+1)/(n_bins)

        fig_height = int(np_max_plot/2*1.5)
        fig, ax = plt.subplots(figsize=(5, fig_height))

        for i in range(n_clusters):
            for j in range(n_bins):
                col_idx = i*n_bins + j
                width = df.iloc[:np_max_plot, col_idx]
                left = df.iloc[:np_max_plot, n_bins*i:col_idx+1].sum(axis=1) - width
                y = y_pos - height*i
                if j == n_bins - 1:
                    # Add label to last bin
                    b = ax.barh(y, width=width, height=height, left=left, 
                                color=color_array[i, j], edgecolor='k', 
                                label=df.columns.tolist()[i*n_bins][0])
                else:
                    b = ax.barh(y, width=width, height=height, left=left, 
                                color=color_array[i, j], edgecolor='k')
                    
                # Increase linewidth for parameters that are sensitive
                for w in enumerate(width.tolist()):
                    if w[1] > 1:
                        b[w[0]].set_linewidth(2.5)
        
        leg = ax.legend()
        # Ensure that linewidths in the legend are all 1.0 pt
        for legobj in leg.legendHandles:
            legobj.set_linewidth(1)

    # Add vertical line and tick labels
    if fmt not in ['indiv', 'bin_avg']:
        ax.axvline(1, color='k', linestyle='--')
    ax.set(yticks=yticks, yticklabels=params[:np_max_plot], xlabel='Sensitivity')
    ax.invert_yaxis()
    
    # Move xaxis label and ticks to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
        
    return fig, ax


def plot_cdf(parameters, labels, parameter, cluster_names=None, colors=None, 
             figsize=(5,5), parameter_names=None, legend_names=None):
    """Plot the class-conditional cdf for a single parameter.
    
    params:
        parameters [array(float)]: array of shape (n_parameters, n_simulations) 
                containing the parameter sets used for each model run
        labels [array(int)]: array of length n_simulations where each value 
                represents the cluster to which that model belongs
        parameter [int]: index of the parameter to plot, corresponding to a 
                column within `parameters`
        cluster_names [list(str)]: ordered list of cluster names corresponding
                to the label array, ie the 0th element of cluster_names is the
                name of the cluster where labels==0. Optional, default is 
                ['Cluster 0', 'Cluster 1', ...]
        colors [list]: colors to plot
        figsize [tuple(float)]: matplotlib figure size in inches. Optional, 
                default: (5,5)
        parameter_name [str]: name of the parameter as listed on the x-axis 
                label. Optional, defaults to 'Parameter #'
        legend_names [list(str)]: ordered list of names to display in the 
                legend. Optional, but must be a permuted version of 
                cluster_names.
                
        
    returns:
        fig: matplotlib figure handle
        ax: matplotlib axis handle
    """
    
    # Check input
    n_clusters = labels.max() + 1
    n_parameters = parameters.shape[1]

    if colors is None:
        colors = []
        cmap = matplotlib.cm.get_cmap('Set1')
        for i in range(n_clusters):
            colors.append(cmap(i))

    if cluster_names is None:
        cluster_names = ['Cluster %s' %i for i in range(n_clusters)]
    
    # If parameter is given as string, look for it in parameter_names
    if isinstance(parameter, str):
        if parameter_names is None:
            raise ValueError('Must provide list of parameter names if providing \
                parameter to plot as a string.')
        elif parameter not in parameter_names:
            raise ValueError('Could not find %s in parameter_names' % parameter)
        else:
            param_idx = parameter_names.index(parameter)
            
    elif isinstance(parameter, int):
        if parameter > (n_parameters - 1):
            raise ValueError('Parameter index passed (%s) is greater than the \
                             length of parameter_names' % parameter)
        else:
            param_idx = parameter
        
    if parameter_names is None:
        parameter_names = ['Parameter %s' %i for i in range(n_parameters)]
    
    # Get list of parameters (used for labeling with full col name)
    percentiles = np.arange(1, 100)

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    for i in range(n_clusters):
        x = np.percentile(parameters[np.where(labels==i), param_idx], percentiles)
        ax.plot(x, percentiles, color=colors[i], label=cluster_names[i])

    ax.set(xlabel=parameter_names[param_idx], ylabel='Percentile')

    # If legend_names is provided, sort the handles before adding legend
    if legend_names is None:
        ax.legend()
    else:
        handles, legend_labels = ax.get_legend_handles_labels()
        reordered_handles = []
        for name in legend_names:
            idx = legend_labels.index(name)
            reordered_handles.append(handles[idx])

        ax.legend(reordered_handles, legend_names)

    return fig, ax
