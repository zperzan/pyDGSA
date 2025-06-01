# -*- coding: utf-8 -*-

"""
pyDGSA sub-package that provides useful utilities for plotting dgsa results.

This file includes:
reformat_interactions
vert_pareto_plot
plot_cdf
calc_bubble_distances
bubble_plot
interaction_matrix
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.manifold import MDS
from tqdm.auto import tqdm


def reformat_interactions(df):
    """Given a dataframe with columns 'k1 | k2', output a dataframe with
    main parameters as the columns and conditional parameters as the rows.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe of sensitivity to interactions

    Returns
    -------
    int_df : dataframe
        Pandas dataframe of sensitivity to interactions, reformatted
        with main parameters as columns and conditional parameters as rows
    """
    for i, interaction in enumerate(df.index.tolist()):
        main_param = interaction.split("|")[0].strip()
        cond_param = interaction.split("|")[1].strip()

        if i == 0:
            int_df = pd.DataFrame(index=[main_param], columns=[cond_param], dtype="float64")
        int_df.loc[cond_param, main_param] = df.loc[interaction, "sensitivity"]

    # Sort df before returning it
    int_df.sort_index(axis=0, inplace=True)
    int_df.sort_index(axis=1, inplace=True)

    return int_df


def vert_pareto_plot(df, np_plot="+5", fmt=None, colors=None, confidence=False, figsize=None):
    """Generate a vertical Pareto plot of sensitivity analysis results.

    Parameters
    ----------
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
                            display cluster's sensitivity separately
                -'bin_avg': bin-specific sensitivities were passed, so display
                            each bin's sensitivity separately
                -'indiv': plots the sensitivity for each bin/cluster combination
                        separately
        confidence [bool]: whether to plot confidence bars. Default is False,
                but must be included in df if confidence == True.
        colors [list(str|int)]: list of clusters colors to use when plotting,
                either specified as rgba tuples or strings of matplotlib named
                colors. Only used when fmt='cluster_avg' or 'indiv'

    Returns
    -------
        fig: matplotlib figure handle
        ax: matplotlib axis handle
    """
    # Total np (number of parameters)
    np_total = df.shape[0]

    # Figure out fmt if not explicitly provided
    if fmt is None:
        if isinstance(df.columns, pd.MultiIndex):
            fmt = "indiv"
        elif df.shape[1] > 1 and "confidence" not in df.columns:
            # Could be either 'cluster_avg' or 'bin_avg'
            if "Cluster" in df.columns[0]:
                fmt = "cluster_avg"
            elif "Bin" in df.columns[0]:
                fmt = "bin_avg"
            else:
                raise ValueError("Could not determine fmt. Please pass explicitly.")
        else:
            # Note that 'mean' also includes 'max' format from the analysis
            fmt = "mean"
            if "confidence" in df.columns:
                cdf = df["confidence"].copy()
                # copy to avoid altering input df
                df = df.drop("confidence", axis=1).copy()

    if fmt == "indiv" and colors is not None:
        if isinstance(colors[0], str):
            # Convert named colors to rgba
            named_colors = colors
            colors = []
            for color in named_colors:
                colors.append(matplotlib.colors.to_rgba(color))

    if fmt == "cluster_avg":
        # Check if confidence bounds were provided by counting
        # columns that end with "_conf"
        conf_cols = [col for col in df.columns if col[-5:] == "_conf"]
        if len(conf_cols) > 0:
            cdf = df[conf_cols].copy()
            df = df.drop(conf_cols, axis=1).copy()

    # Get number of parameters with sensitivity >= 1
    na_mask = np.isnan(df.astype(float).values)
    np_sensitive = np.sum(df.values[~na_mask] >= 1)

    # Figure out how many parameters to plot
    if isinstance(np_plot, str):
        if np_plot == "all":
            np_max_plot = np_total
        elif np_plot[0] == "+":
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

    if fmt == "mean" or fmt == "max":
        # Sort so most sensitive params are on top
        df.sort_values(by=df.columns[0], ascending=False, inplace=True)
        data = df.values[:np_max_plot, :].squeeze()
        params = df.index.tolist()  # Get list of params after sorting

        yticks = y_pos

        # Error bars (confidence interval); by default these are plotted, but of
        # length 0 if confidence == False
        if confidence:
            xerr = cdf[df.index[:np_max_plot]].values / 2
        else:
            xerr = 0

        # Values are color-coded. If confidence intervals are provided
        if confidence:
            colors = np.asarray([[1, 1, 1, 0.8]] * np_max_plot)
            # > confidence interval red
            colors[data - xerr > 1] = [1, 0, 0, 0.8]
            # < confidence interval blue
            colors[data + xerr < 1] = [0, 0, 1, 0.8]
        else:
            if np_sensitive > 0:
                colors = np.asarray([[1, 0, 0, 0.8]] * np_max_plot)
            else:
                colors = np.asarray([[0, 0, 1, 0.8]] * np_max_plot)

            if np_max_plot > np_sensitive:
                if np_sensitive > 0:
                    colors[np_sensitive] = [1, 1, 1, 1]
                colors[np_sensitive + 1 :] = [0, 0, 1, 0.8]

        if figsize is None:
            fig_height = int(np_max_plot / 2)
            figsize = (5, fig_height)

        # Create figure and add barh
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(y_pos, data, color=colors, edgecolor="k", xerr=xerr)

    elif fmt == "cluster_avg":
        n_clusters = df.shape[1]

        # Sort by mean sensitivity across clusters
        sort_df = df.mean(axis=1).sort_values(ascending=False)
        df = df.reindex(sort_df.index)
        params = df.index.tolist()  # Get list of params after sorting

        # Add error bars if confidence=True, otherwise set length to 0
        if confidence:
            xerr = cdf.loc[df.index, :].values
        else:
            xerr = df.loc[df.index, :].values * 0

        height = 1 / (n_clusters + 1)
        yticks = y_pos - (height * (n_clusters - 1) / 2)

        if colors is None:
            colors = []
            cmap = plt.colormaps.get_cmap("Set1")
            for i in range(n_clusters):
                colors.append(cmap(i))

        # Create figure
        if figsize is None:
            fig_height = int(np_max_plot / 2 * 1.5)
            figsize = (5, fig_height)

        fig, ax = plt.subplots(figsize=figsize)

        # Add bars for each cluster
        for i in range(n_clusters):
            ax.barh(
                y_pos - height * i,
                df.iloc[:np_max_plot, i],
                height=height,
                color=colors[i],
                edgecolor="k",
                label=df.columns.tolist()[i],
                xerr=xerr[:np_max_plot, i],
            )
        ax.legend()

    elif fmt == "bin_avg":
        n_bins = df.shape[1]

        # Sort by mean sensitivity across bins
        sort_df = df.mean(axis=1).sort_values(ascending=False)
        df = df.reindex(sort_df.index)
        params = df.index.tolist()

        yticks = y_pos

        if colors is None:
            cmap = plt.colormaps.get_cmap("Set1")
            colors = cmap(1)

        # Create color array by decreasing alpha channel for each bin
        color_array = np.tile(colors, (n_bins, 1))
        for i in range(n_bins):
            color_array[i, 3] = (i + 1) / (n_bins)

        if figsize is None:
            fig_height = int(np_max_plot / 2 * 1.5)
            figsize = (5, fig_height)

        fig, ax = plt.subplots(figsize=figsize)

        for i in range(n_bins):
            width = df.iloc[:np_max_plot, i]
            left = df.iloc[:np_max_plot, :i].sum(axis=1)
            b = ax.barh(y_pos, width=width, left=left, color=color_array[i], edgecolor="k")

            # Increase linewidth for parameters that are sensitive
            for w in enumerate(width.tolist()):
                if w[1] > 1:
                    b[w[0]].set_linewidth(2.5)

    elif fmt == "indiv":
        n_clusters, n_bins = df.columns.levshape

        # Split df into sensitive and non-sensitive df's, sort each, then re-combine
        # Can't just sort on mean sensitivity, otherwise a sensitive parameter
        # could get left out because its mean might not be within the top
        # np_max_plot most sensitive, even though a single bin is >= 1
        sens_arr = df.copy()
        na_mask = np.isnan(df.astype(float).values)
        sens_arr[na_mask] = 0
        sens_mask = np.any(sens_arr >= 1, axis=1)
        sens = df[sens_mask].copy()
        nsens = df[~sens_mask].copy()
        sort_sens = sens.mean(axis=1).sort_values(ascending=False)
        sens = sens.reindex(sort_sens.index)
        sort_nsens = nsens.mean(axis=1).sort_values(ascending=False)
        nsens = nsens.reindex(sort_nsens.index)
        df = pd.concat((sens, nsens))

        params = df.index.tolist()
        height = 1 / (n_clusters + 1)
        yticks = y_pos - (height * (n_clusters - 1) / 2)

        if colors is None:
            cmap = plt.colormaps.get_cmap("Set1")
            colors = []
            for i in range(n_clusters):
                colors.append(cmap(i))

        # Create color array by decreasing alpha channel for each bin
        color_array = np.zeros((n_clusters, n_bins, 4), dtype="float64")
        for i in range(n_clusters):
            for j in range(n_bins):
                color_array[i, j, :] = colors[i]
                color_array[i, j, 3] = (j + 1) / (n_bins)

        fig_height = int(np_max_plot / 2 * 1.5)
        fig, ax = plt.subplots(figsize=(5, fig_height))

        for i in range(n_clusters):
            for j in range(n_bins):
                col_idx = i * n_bins + j
                width = df.iloc[:np_max_plot, col_idx]
                left = df.iloc[:np_max_plot, n_bins * i : col_idx + 1].sum(axis=1) - width
                y = y_pos - height * i
                if j == n_bins - 1:
                    # Add label to last bin
                    b = ax.barh(
                        y,
                        width=width,
                        height=height,
                        left=left,
                        color=color_array[i, j],
                        edgecolor="k",
                        label=df.columns.tolist()[i * n_bins][0],
                    )
                else:
                    b = ax.barh(y, width=width, height=height, left=left, color=color_array[i, j], edgecolor="k")

                # Increase linewidth for parameters that are sensitive
                for w in enumerate(width.tolist()):
                    if w[1] > 1:
                        b[w[0]].set_linewidth(2.5)

        leg = ax.legend()
        # Ensure that linewidths in the legend are all 1.0 pt
        for legobj in leg.legend_handles:
            legobj.set_linewidth(1)

    # Add vertical line and tick labels
    if fmt not in ["indiv", "bin_avg"]:
        ax.axvline(1, color="k", linestyle="--")
    ax.set(yticks=yticks, yticklabels=params[:np_max_plot], xlabel="Sensitivity")
    ax.invert_yaxis()

    # Move xaxis label and ticks to top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    return fig, ax


def get_param_idx(parameter, parameter_names, n_parameters):
    """Regardless of format given, get the index of a specific parameter.

    Parameters
    ----------
    parameter : str
        parameter of interest
    parameter_names : list
        A list of parameter names within the dataframe
    n_parameters: int
        Total number of parameters

    Returns
    -------
    param_idx : int
        Index of parameter within parameter_names
    """
    # If parameter is given as string, look for it in parameter_names
    if isinstance(parameter, str):
        if parameter_names is None:
            raise ValueError(
                "Must provide list of parameter names if providing \
                parameter to plot as a string."
            )
        elif parameter not in parameter_names:
            raise ValueError("Could not find %s in parameter_names" % parameter)
        else:
            param_idx = parameter_names.index(parameter)

    elif isinstance(parameter, int):
        if parameter > (n_parameters - 1):
            raise ValueError(
                "Parameter index passed (%s) is greater than the \
                             length of parameter_names"
                % parameter
            )
        else:
            param_idx = parameter

    return param_idx


def plot_interact_cdf(
    parameters,
    labels,
    main_param,
    cond_param,
    cluster_names=None,
    colors=None,
    figsize=(12, 5),
    parameter_names=None,
    n_bins=3,
    bin_labels=None,
    plot_prior=True,
):
    """Plot the cluster-conditional CDF of a parameter, conditioned to another parameter.

    Parameters
    ----------
    parameters : numpy array
        Parameter values for each model simmulation, with rows corresponding to the number
        of simulations and columns corresponding to the number of parameters
    labels : numpy array
        Labels of the response cluster to which each simulation belongs. Should be a 1d
        array with rows equal to the number of simulations
    main_param : str
        The main parameter for which to plot the conditional CDF
    cond_param : str
        The parameter on which the CDF will be conditioned
    cluster_names : list
        The names of each response cluster included in labels. Optional, defaults
        to ['Cluster 0', 'Cluster 1', etc.]
    colors : list
        Colors corresponding to each response cluster. Optional, defaults to 'Set 1'
    figsize : tuple
        Figure size in inches. Optional, defaults to (12, 5)
    parameter_names : list
        A list of parameter names. Optional, defaults to ['Parameter 0', 'Parameter 1',
        'Parameter 2', etc.]
    n_bins : int
        Number of bins to divide the conditional parmaeter into. Optional, defaults to 3
    bin_labels : list
        Labels for each bin for the conditional parameter. Optional, defaults to ['Bin 0',
        'Bin 1', 'Bin 2', etc.]
    plot_prior : bool
        Whether to plot the distribution of the parameter across all simulations. Optional,
        defaults to True

    Returns
    -------
    fig: matplotlib figure handle
        Matplotlib figure object
    ax: matplotlib axis handle
        Matplotlib axes object
    """
    # Check input
    n_clusters = labels.max() + 1
    n_parameters = parameters.shape[1]

    if colors is None:
        colors = []
        cmap = plt.colormaps.get_cmap("Set1")
        for i in range(n_clusters):
            colors.append(cmap(i))

    # Create color array by decreasing alpha channel for each bin
    color_array = np.zeros((n_clusters, n_bins, 4), dtype="float64")
    for i in range(n_clusters):
        for j in range(n_bins):
            color_array[i, j, :] = colors[i]
            color_array[i, j, 3] = (j + 1) / (n_bins)

    if cluster_names is None:
        cluster_names = ["Cluster %s" % i for i in range(n_clusters)]

    if bin_labels is None:
        bin_labels = ["Bin %s" % i for i in n_bins]

    # Calculate the thresholds separating bins for each parameter
    # thresholds is of shape (n_bins, n_parameters)
    thresholds = np.quantile(parameters, [b / n_bins for b in range(1, n_bins)], axis=0)

    # Get the indices of the main and cond params
    mpidx = get_param_idx(main_param, parameter_names, n_parameters)
    cpidx = get_param_idx(cond_param, parameter_names, n_parameters)

    if parameter_names is None:
        parameter_names = ["Parameter %s" % i for i in range(n_parameters)]

    percentiles = np.arange(1, 100)

    fig, ax = plt.subplots(1, n_clusters, figsize=figsize, tight_layout=True, sharey=True)

    for nc in range(n_clusters):
        c_idx = np.where(labels == nc)
        q_prior = np.percentile(parameters[c_idx, mpidx], percentiles)
        if plot_prior:
            ax[nc].plot(
                np.percentile(parameters[:, mpidx], percentiles),
                percentiles,
                color="k",
                linestyle="--",
                label="$F$ ($X_i$)",
            )
        ax[nc].plot(
            q_prior, percentiles, color=color_array[nc, -2], linestyle=":", label="$F$ ($X_i$ | %s)" % cluster_names[nc]
        )

        for nb in range(n_bins):
            # For each bin, first calc b_mask -- mask of params within each bin
            # The first bin
            if nb == 0:
                threshold = thresholds[nb, cpidx]
                b_mask = parameters[:, cpidx] <= threshold
            # The last bin
            elif nb == (n_bins - 1):
                threshold = thresholds[-1, cpidx]
                b_mask = parameters[:, cpidx] > threshold
            # Middle bins
            else:
                low_thresh = thresholds[nb - 1, cpidx]
                high_thresh = thresholds[nb, cpidx]
                b_mask = (parameters[:, cpidx] > low_thresh) & (parameters[:, cpidx] <= high_thresh)

            # Indices comprising this bin
            b_idx = np.argwhere(b_mask)

            # Indices within this cluster AND bin
            bc_idx = np.intersect1d(c_idx, b_idx.flatten(), assume_unique=True)
            if len(bc_idx) > 2:
                q_inter = np.percentile(parameters[bc_idx, mpidx], percentiles, axis=0)
                label = "$F$ ($X_i$ | %s, %s)" % (cluster_names[nc], bin_labels[nb])
                ax[nc].plot(q_inter, percentiles, label=label, color=color_array[nc, nb])

        ax[nc].set(
            xlim=[parameters[:, mpidx].min(), parameters[:, mpidx].max()],
            xlabel=parameter_names[mpidx],
            title=cluster_names[nc],
        )
        ax[nc].legend()
    ax[0].set(ylabel="% of simulations")

    return fig, ax


def calc_bubble_distances(int_df):
    """Given a dataframe of asymmetric interactions, calculate
    the symmetric distance for a bubble plot, as proposed by Park
    et al. (2016)

    Parameters
    ----------
    int_df : dataframe
        Pandas dataframe of interactions between parameters, with columns
        using the format ['p1 | p2', 'p1 | p3', etc.]

    Returns
    -------
    sym_df : dataframe
        Pandas dataframe containing the symmetric distance between parameter
        iteractions as describe by Park et al. (2016)
    """
    ref_df = reformat_interactions(int_df)
    sym_df = ref_df.copy()

    # Drop main_params that are not in cond_params
    for main_param in ref_df.columns:
        if main_param not in ref_df.index:
            sym_df.drop(main_param, axis=1, inplace=True)

    for cond_param in sym_df.index:
        for main_param in sym_df.columns:
            if main_param == cond_param:
                sym_df.loc[cond_param, main_param] = 0
            else:
                int1 = ref_df.loc[cond_param, main_param]
                int2 = ref_df.loc[main_param, cond_param]
                sym_df.loc[cond_param, main_param] = 2 / (int1 + int2)

    return sym_df


def bubble_plot(main_df, int_df, figsize=(5, 5)):
    """Generate a bubble plot of interactions between parameters. Within the
    plot, the size of a bubble corresponds to the sensitivity of that parameter
    alone, while the proximity between bubbles corresponds to those parameters'
    interactions.

    Parameters
    ----------
    main_df : dataframe
        Main effects calculated by DGSA
    int_df : dataframe
        Interaction effects calculated by DGSA
    figsize : tuple
        Figure size in inches. Optional, default: (5,5)

    Returns
    -------
    fig: matplotlib figure handle
        Matplotlib figure object
    ax: matplotlib axis handle
        Matplotlib axes object
    """
    sym_df = calc_bubble_distances(int_df)

    # Calculate MDS coordinates of the
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0)
    mds_dist = mds.fit_transform(sym_df.values)

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    x = mds_dist[:, 0]
    y = mds_dist[:, 1]
    sizes = main_df.loc[sym_df.index, "sensitivity"] * 1000
    colors = []
    for sens in main_df.loc[sym_df.index, "sensitivity"]:
        if sens > 1:
            colors.append("red")
        else:
            colors.append("blue")

    ax.scatter(x, y, s=sizes, c=colors, alpha=0.5)

    for i, label in enumerate(sym_df.index):
        ax.annotate(label, (x[i], y[i]))

    # Remove tick labels
    ax.set(yticklabels=[], xticklabels=[])

    return fig, ax


def interaction_matrix(df, figsize=(5, 5), fontsize=12, nan_color="gray"):
    """Given a dataframe of asymmetric sensitivities, plot a color-coded
    matrix of those sensitivities.

    Parameters
    ----------
    df : dataframe
        Pandas dataframe containing the sensitivity analysis results.
    figsize : tuple
        Matplotlib figure size in inches. Optional, default: (5,5)
    fontsize : float
        Font size for the text values in the matrix. Optional, default: 12
    nan_color : str
        A matplotlib named color for displaying NaN values. Optional, default
        is 'gray'

    Returns
    -------
    fig: matplotlib figure handle
        Matplotlib figure object
    ax: matplotlib axis handle
        Matplotlib axes object
    """
    int_df = reformat_interactions(df)
    main_params = int_df.columns.tolist()
    cond_params = int_df.index.tolist()

    # Separate out sensitive and insensitive parameters
    # and plot using separate color ramps
    mask = int_df.values > 1

    sens = int_df.values.copy()
    insens = int_df.values.copy()
    sens[~mask] = np.nan
    insens[mask] = np.nan

    # Create separate array of NaN's as nan_color
    nan_rgb = matplotlib.colors.to_rgba(nan_color)  # convert str to rgb values
    nancells = np.tile(int_df.values.copy(), (4, 1, 1))

    # Swap all NaN's to RGBA and everything else to 0
    nancells[~np.isnan(nancells)] = 0
    for i, cvalue in enumerate(nan_rgb):
        nancells[i, np.isnan(nancells[i, :, :])] = cvalue

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(insens, vmin=0, vmax=1.3, cmap="Blues", interpolation="none")
    if not np.all(np.isnan(sens)):
        ax.imshow(sens, vmin=0.7, vmax=np.nanmax(sens) * 1.5, cmap="Reds", interpolation="none")
    ax.imshow(nancells.swapaxes(0, 2).swapaxes(0, 1))

    ax.set(xticks=range(len(main_params)), yticks=range(len(cond_params)), yticklabels=cond_params)

    # Add rotated xtick labels
    ax.set_xticklabels(labels=main_params, rotation=90)

    for i, main_param in enumerate(main_params):
        for j, cond_param in enumerate(cond_params):
            value = int_df.loc[cond_param, main_param]
            if not np.isnan(value):
                text = "%0.2f" % value
                ax.text(i, j, text, fontsize=fontsize, horizontalalignment="center", verticalalignment="center")

    # Label axes
    ax.set(xlabel="Main parameter", ylabel="Cond. parameter")

    return fig, ax


def plot_cdf(
    parameters,
    labels,
    parameter,
    cluster_names=None,
    colors=None,
    figsize=(5, 5),
    parameter_names=None,
    legend_names=None,
    plot_prior=False,
    plot_boots=False,
    n_boots=300,
    progress=False,
):
    """Plot the class-conditional cdf for a single parameter.

    Parameters
    ----------
    parameters : ndarray
        An array of shape (n_parameters, n_simulations) containing the parameter
        sets used for each model run
    labels : ndarray
        An array of length n_simulations where each value represents the cluster
        to which that model belongs
    parameter : int
        Index of the parameter to plot, corresponding to a column within the
        `parameters` dataframe
    cluster_names : list
        Ordered list of cluster names corresponding to the label array, ie the
        0th element of cluster_names is the name of the cluster where labels==0.
        Optional, default is ['Cluster 0', 'Cluster 1', ...]
    colors : list
        colors corresponding to each cluster. Optional, default is colors from
        the matplotlib 'Set 1' colormap
    figsize : tuple
        Matplotlib figure size in inches. Optional, default: (5,5)
    parameter_names: list
        Names of the parameter as listed on the x-axis label. Optional,
        defaults to ['Parameter 0', 'Parameter 2', etc.]
    legend_names : list
        Ordered list of names to display in the legend. Optional, but must be
        a permuted version of cluster_names.
    plot_prior : bool
        whether to plot the prior distribution of the parameter. Optional,
        defaults to False.
    plot_boots : bool
        Whether to plot the bootstrapped distributions of the parameter
        corresponding to each cluster. Note that this option re-calculates the
        bootstrapped distributions simply for plotting purposes, and they are
        not the same as the bootstrapped datasets used in the `dgsa` function.
        Optional, defaults to False.
    n_boots : int
        Number of bootstrap samples to use if `plot_boots` is True. Optional,
        defaults to 300.
    progress : bool
        Whether to show a progress bar when calculating bootstrapped distributions.
        Optional, defaults to False.

    Returns
    -------
    fig: matplotlib figure handle
        Matplotlib figure object
    ax: matplotlib axis handle
        Matplotlib axes object
    """
    # Check input
    n_clusters = labels.max() + 1
    n_parameters = parameters.shape[1]

    if colors is None:
        colors = []
        cmap = plt.colormaps.get_cmap("Set1")
        for i in range(n_clusters):
            colors.append(cmap(i))

    if cluster_names is None:
        cluster_names = ["Cluster %s" % i for i in range(n_clusters)]

    # If parameter is given as string, look for it in parameter_names
    if isinstance(parameter, str):
        if parameter_names is None:
            raise ValueError(
                "Must provide list of parameter names if providing \
                parameter to plot as a string."
            )
        elif parameter not in parameter_names:
            raise ValueError("Could not find %s in parameter_names" % parameter)
        else:
            param_idx = parameter_names.index(parameter)

    elif isinstance(parameter, int):
        if parameter > (n_parameters - 1):
            raise ValueError(
                "Parameter index passed (%s) is greater than the \
                             length of parameter_names"
                % parameter
            )
        else:
            param_idx = parameter

    if parameter_names is None:
        parameter_names = ["Parameter %s" % i for i in range(n_parameters)]

    # Get list of parameters (used for labeling with full col name)
    percentiles = np.arange(1, 100)

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

    if plot_boots:
        added_label = False
        if progress:
            iterator = tqdm(range(n_boots))
        else:
            iterator = range(n_boots)
        for nb in iterator:
            for i in range(n_clusters):
                cluster_size = np.sum(labels == i)
                boot_idx = np.random.choice(np.arange(parameters.shape[0]), size=cluster_size, replace=False)
                x = np.percentile(parameters[boot_idx, param_idx], percentiles)
                if not added_label:
                    ax.plot(x, percentiles, color="k", alpha=0.1, label="$F$ ($X_i$) bootstrapped")
                    added_label = True
                else:
                    ax.plot(x, percentiles, color="0.5", alpha=0.1)

    if plot_prior:
        ax.plot(
            np.percentile(parameters[:, param_idx], percentiles),
            percentiles,
            color="k",
            linestyle="--",
            label="$F$ ($X_i$)",
        )

    for i in range(n_clusters):
        x = np.percentile(parameters[np.where(labels == i), param_idx], percentiles)
        ax.plot(x, percentiles, color=colors[i], label="$F$ ($X_i$ | %s)" % cluster_names[i])

    ax.set(xlabel=parameter_names[param_idx], ylabel="% of simulations")

    # If legend_names is provided, sort the handles before adding legend
    if legend_names is None:
        ax.legend()
    else:
        handles, legend_labels = ax.get_legend_handles_labels()
        reordered_handles = []
        for name in legend_names:
            idx = legend_labels.index("$F$ ($X_i$ | %s)" % name)
            reordered_handles.append(handles[idx])

        ax.legend(reordered_handles, legend_names)

    return fig, ax


def plot_pdf(
    parameters,
    labels,
    parameter,
    cluster_names=None,
    colors=None,
    figsize=(5, 5),
    parameter_names=None,
    legend_names=None,
    plot_prior=False,
    models_per_bin=5,
    **kwargs,
):
    """Plot the class-conditional pdf for a single parameter.

    Parameters
    ----------
    parameters : ndarray
        An array of shape (n_parameters, n_simulations) containing the
        parameter sets used for each model run
    labels : ndarray
        An aarray of length n_simulations where each value represents the
        cluster to which that model belongs
    parameter : int
        Index of the parameter to plot, corresponding to a column within the
        `parameters` dataframe
    cluster_names : list
        Ordered list of cluster names corresponding to the label array, ie
        the 0th element of cluster_names is the name of the cluster where
        labels==0. Optional, default is ['Cluster 0', 'Cluster 1', ...]
    colors : list
        colors corresponding to each cluster. Optional, default is colors from
        the matplotlib 'Set 1' colormap
    figsize : tuple
        Matplotlib figure size in inches. Optional, default: (5,5)
    parameter_names: list
        Names of the parameter as listed on the x-axis label. Optional,
        defaults to ['Parameter 0', 'Parameter 2', etc.]
    legend_names : list
        Ordered list of names to display in the legend. Optional, but must be
        a permuted version of cluster_names.
    plot_prior : bool
        whether to plot the prior distribution of the parameter. Optional,
        defaults to False.
    models_per_bin : int
        This defines the spacing of the x-values in the pdf plot. A larger
        number means larger bins, which means lower resolution along the x-axis.
        Optional, default: 5
    **kwargs : kwargs
        keyword arguments to pass to scipy.stats.gaussian_kde

    Returns
    -------
    fig: matplotlib figure handle
        Matplotlib figure object
    ax: matplotlib axis handle
        Matplotlib axes object
    """
    # Check input
    n_clusters = labels.max() + 1
    n_parameters = parameters.shape[1]

    if colors is None:
        colors = []
        cmap = plt.colormaps.get_cmap("Set1")
        for i in range(n_clusters):
            colors.append(cmap(i))

    if cluster_names is None:
        cluster_names = ["Cluster %s" % i for i in range(n_clusters)]

    # If parameter is given as string, look for it in parameter_names
    if isinstance(parameter, str):
        if parameter_names is None:
            raise ValueError(
                "Must provide list of parameter names if providing \
                parameter to plot as a string."
            )
        elif parameter not in parameter_names:
            raise ValueError("Could not find %s in parameter_names" % parameter)
        else:
            param_idx = parameter_names.index(parameter)

    elif isinstance(parameter, int):
        if parameter > (n_parameters - 1):
            raise ValueError(
                "Parameter index passed (%s) is greater than the \
                             length of parameter_names"
                % parameter
            )
        else:
            param_idx = parameter

    # Get list of parameters (used for labeling with full col name)
    if parameter_names is None:
        parameter_names = ["Parameter %s" % i for i in range(n_parameters)]

    fig, ax = plt.subplots(figsize=figsize, tight_layout=True, facecolor="white")

    if plot_prior:
        n = int(labels.shape[0] / models_per_bin)
        pmin = parameters[:, param_idx].min()
        pmax = parameters[:, param_idx].max()
        prange = pmax - pmin
        pmin -= prange * 0.2
        pmax += prange * 0.2
        x = np.linspace(pmin, pmax, n)
        kernel = gaussian_kde(parameters[:, param_idx].flatten(), **kwargs)
        pdf = kernel(x)
        ax.fill_between(x, 0, pdf, alpha=0.6, label="$f$ ($X_i$)", color="k", zorder=1)
        ax.plot(x, pdf, color="k", alpha=0.8, zorder=1)

    for i in range(n_clusters):
        n = int((labels == i).sum() / models_per_bin)
        label = "$f$ ($X_i$ | %s)" % cluster_names[i]
        pmin = parameters[np.where(labels == i), param_idx].min()
        pmax = parameters[np.where(labels == i), param_idx].max()
        prange = pmax - pmin
        pmin -= prange * 0.2
        pmax += prange * 0.2
        x = np.linspace(pmin, pmax, n)
        kernel = gaussian_kde(parameters[np.where(labels == i), param_idx].flatten(), **kwargs)
        pdf = kernel(x)

        ax.fill_between(x, 0, pdf, alpha=0.6, label=label, color=colors[i], zorder=1 + ((i + 1) / n_clusters))
        ax.plot(x, pdf, color=colors[i], alpha=0.8, zorder=1.05 + ((i + 1) / n_clusters))

    ax.set(xlabel=parameter_names[param_idx], ylabel="Density")

    # If legend_names is provided, sort the handles before adding legend
    if legend_names is None:
        ax.legend()
    else:
        handles, legend_labels = ax.get_legend_handles_labels()
        reordered_handles = []
        for name in legend_names:
            idx = legend_labels.index("$F$ ($X_i$ | %s)" % name)
            reordered_handles.append(handles[idx])

        ax.legend(reordered_handles, legend_names)

    return fig, ax
