# -*- coding: utf-8 -*-

"""
Main pyDGSA module that provides the following functions:

dgsa
dgsa_interactions
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .interact_util import interact_boot_distance, interact_distance


def cluster_cdf_distance(prior_cdf, cluster_parameters, percentiles=None):
    """Calculate the L1-norm distance between the prior distribution of each
    parameter and the class-conditional distribution of each parameter. In
    other words, the distance between the cdf of a parameter for all model
    runs and the cdf of a parameter for model runs within an individual cluster.

    Parameters
    ----------
    prior_cdf : ndarray
        The prior cdf of the parameter
    cluster_parameters : ndarray
        An array of parameter values in an individual cluster
    percentiles : ndarray
        The percentile values for which `prior_cdf` was calculated. Optional,
        defaults to `np.arange(1,100)`

    Returns
    -------
    cluster_distance : float
        The L1-norm distance between the prior cdf and the cluster-conditional
        cdf
    """
    if percentiles is None:
        percentiles = np.arange(1, 100)

    # Calculate the cdf of parameters in this individual cluster
    cluster_cdf = np.percentile(cluster_parameters, percentiles, axis=0)

    # Calculate the L1-norm between the cdf of parameters in this cluster
    # and the cdf of the entire population
    cluster_distance = np.sum(abs(cluster_cdf - prior_cdf), axis=0)

    return cluster_distance


def dgsa(
    parameters,
    labels,
    parameter_names=None,
    n_boots=3000,
    quantile=0.95,
    output="mean",
    cluster_names=None,
    confidence=False,
    progress=True,
    replace=False,
):
    """Given a parameter set and clustered model responses, calculate the
    normalized model sensitivity to each parameter, without interactions.

    Parameters
    ----------
    parameters :
        An array of shape (n_parameters, n_simulations) containing the parameter
        sets used for each model run
    labels : ndarray
        An array of length n_simulations where each value represents the cluster
        to which that model belongs.
    parameter_names : list
        The list of parameter names. Optional, defaults to ['param0',
        'param1', ..., 'paramN']
    n_boots : int
        The number of bootstrapped datasets to create for hypothesis testing.
        Optional, defaults to 3000
    quantile : float
        Quantile used to test the null hypothesis. Can specify as a percentile
        [0-100] or quantile [0-1]. Optional, defaults to 0.95
    confidence : bool
        Whether to output confidence intervals on the standardized sensitivity measure,
        following Park et al. (2016).
    output : str
        Whether to return the mean standardized sensitivity across clusters ('mean'),
        the max sensitivity across clusters ('max'), or the standardized sensitivity
        for each cluster ('cluster_avg').
    cluster_names : list
        The list of cluster names. Optional, defaults to ['Cluster 0', 'Cluster 1',
        ..., 'Cluster N']
    progress : bool
        Whether to display a tqdm progress bar during calculation. Optional, default
        is True.
    replace : bool
        Whether to sample with replacement when generating bootstrapped datasets.
        Optional, default is False. Note that the original DGSA package (Park et
        al., Computers & Geosciences, 2016) did not use replacement when sampling,
        so the default option here is consistent with that original package. See:
        https://github.com/SCRFpublic/DGSA/blob/f781012291c43e00e1ffea185abfec6ef852b0ab/DGSA_computations/BootstrapMainFactors.m#L41

    Returns
    -------
    df : dataframe
        Pandas dataframe containing the normalized sensitivity of each parameter.
        If output='mean' or output='max', indices are parameter names and the single
        column is 'sensitivity'. If output='cluster_avg', indices are the parameter
        names and each column is an individual cluster.
    """
    ### Check the input and convert to dict
    n_clusters = labels.max() + 1
    n_parameters = parameters.shape[1]

    # Generate default parameter names if not provided
    if parameter_names is None:
        parameter_names = ["param{}".format(i) for i in range(n_parameters)]

    if output not in ["cluster_avg", "mean", "max"]:
        raise ValueError("Requested output format must be 'cluster_avg', 'mean' or 'max'")

    # Convert label array to dict of cluster indices
    # Also check that there are at least 10 models per class [Fenwick et al., 2014]
    clusters = {}
    for nc in range(n_clusters):
        cluster = np.where(labels == nc)[0]
        clusters[nc] = cluster
        if len(cluster) < 10:
            print(
                "Warning: \
                  Cluster {} contains {} models. \
                  Recommend >=10 models per cluster for optimal performance.".format(nc, len(cluster))
            )

    ### Step 1.
    # Caclulate distances between the prior cdfs and the cluster-conditional cdf

    # Calculate the prior cdf
    percentiles = np.arange(1, 100)
    prior_cdf = np.percentile(parameters, percentiles, axis=0)

    # Iterate over each cluster
    cluster_distances = np.zeros((n_clusters, n_parameters))
    for i, cluster in clusters.items():
        # Get the model parameters for this cluster
        cluster_parameters = parameters[cluster]
        cluster_distances[i, :] = cluster_cdf_distance(prior_cdf, cluster_parameters, percentiles=percentiles)

    ### Step 2.
    # Now that we have the distances for each cluster, test for statistical
    # significance using bootstrapping
    boot_distances = np.zeros((n_boots, n_clusters, n_parameters))

    # Loop through n_boots
    if progress:
        iterator = tqdm(range(n_boots))
    else:
        iterator = range(n_boots)
    for nb in iterator:
        # For each boot, generate random clusters the same size as our clusters
        # and calc cdf distance between those
        for i, cluster in clusters.items():
            # Remove extra cluster dim if provided (prevents shape mismatch error)
            cluster = cluster.squeeze()

            # Generate random indices of size equal to that of the current cluster
            boot_idx = np.random.choice(np.arange(parameters.shape[0]), size=len(cluster), replace=replace)
            boot_parameters = parameters[boot_idx]

            # Calculate cdf distance for this random cluster
            boot_distances[nb, i, :] = cluster_cdf_distance(prior_cdf, boot_parameters, percentiles=percentiles)

    # Get the nth quantile of all boots
    boot_quantiles = np.quantile(boot_distances, quantile, axis=0)

    ### Step 3.
    # Calculate the normalized distances (d/d_95)
    cluster_sensitivities = cluster_distances / boot_quantiles

    # Calculate the confidence interval, if requested
    if confidence:
        dist_upper = np.quantile(boot_distances, 1, axis=0)
        dist_lower = np.quantile(boot_distances, quantile - (1 - quantile), axis=0)
        upper_bound = cluster_distances / dist_upper
        lower_bound = cluster_distances / dist_lower

    # Output as a dataframe
    if output == "mean":
        columns = ["sensitivity"]
        data = np.mean(cluster_sensitivities, axis=0)

        if confidence:
            conf = np.mean(lower_bound - upper_bound, axis=0)
            columns.append("confidence")
            data = np.column_stack((data, conf))

        df = pd.DataFrame(data, index=parameter_names, columns=columns)

    elif output == "max":
        columns = ["sensitivity"]
        data = np.max(cluster_sensitivities, axis=0)

        if confidence:
            conf = np.max(lower_bound - upper_bound, axis=0)
            columns.append("confidence")
            data = np.column_stack((data, conf))

        df = pd.DataFrame(data, index=parameter_names, columns=columns)

    elif output == "cluster_avg":
        if cluster_names is None:
            cluster_names = ["Cluster " + str(i) for i in range(n_clusters)]
        columns = cluster_names.copy()

        data = cluster_sensitivities.T

        if confidence:
            conf = lower_bound.T - upper_bound.T
            for i, cluster in enumerate(cluster_names):
                columns.insert(2 * i + 1, cluster + "_conf")
                data = np.insert(data, 2 * i + 1, conf[:, i], axis=1)

        df = pd.DataFrame(data, index=parameter_names, columns=columns)

    return df


def dgsa_interactions(
    parameters,
    labels,
    cond_parameters=None,
    n_bins=3,
    parameter_names=None,
    quantile=0.95,
    n_boots=3000,
    output="mean",
    cluster_names=None,
    progress=True,
):
    """Given a parameter set, clustered model responses, and a list of
    conditional parameters, calculate the sensitivity of each two-way
    parameter interaction.

    Parameters
    ----------
    parameters : ndarray
        An array of shape (n_parameters, n_simulations) containing the
        parameter sets used for each model run
    labels : ndarray
        An array of length n_simulations where each value represents the
        cluster to which that model belongs.
    cond_parameters : list
        A list of conditional parameters for which to calculate interactions.
        Can be either a list of indices corresponding to columns of parameters,
        or a list of names corresponding to parameter_names. Optional, defaults
        to calculating interactions for all parameters
    n_bins : int
        Number of bins in which to separate out each conditional parameter
    parameter_names : list
        Ordered list of parameter names matching column order within parameters.
        Optional, defaults to ['param0', 'param1', ..., 'paramN']
    n_boots : int
        The number of bootstrapped datasets to create for hypothesis testing.
        Optional, defaults to 3000
    quantile : float
        Quantile used to test the null hypothesis. Can specify as a percentile
        [0-100] or quantile [0-1]. Optional, defaults to 0.95
    output : str
        Format in which to return sensitivities. Must be either 'mat', 'mean',
        'cluster_avg', 'bin_avg', or 'indiv'. 'mat' returns sensitivities in
        matrix form (main params as columns and conditional params as indices).
        'mean' returns in single column form with each individual interaction
        spelled out in the indices (e.g., 'x | y'). 'cluster_avg' returns the
        average sensitivity for each cluster, without bin-specific values.
        'bin_avg' returns the average sensitivity for each bin, without
        cluster-specific values. 'indiv' returns the sensitivity for each
        bin/cluster combination and no averaging is performed. Optional, defaults
        to 'mean'.
    cluster_names : list
        A list of cluster names. Optional, defaults to ['Cluster 0', 'Cluster 1',
        ..., 'Cluster N']
    progress : bool
        Whether to display a tqdm progress bar during calculation. Optional,
        default is True.

    Returns
    -------
    df : dataframe
        pandas dataframe containing the normalized sensitivity of each two-way
        parameter interaction. Columns correspond to each parameter in parameter_names
        and indices are the conditonal parameters in cond_parameters.
    """
    # Get number of clusters, parameters, etc
    n_parameters = parameters.shape[1]
    n_clusters = labels.max() + 1

    # Generate default list of parameter names
    if parameter_names is None:
        parameter_names = ["param{}".format(i) for i in range(n_parameters)]

    if cond_parameters is None:
        cond_parameters = parameter_names

    if output not in ["mat", "mean", "cluster_avg", "bin_avg", "indiv"]:
        raise ValueError("Output format must be 'mean', 'mat', 'indiv', 'bin_avg', or 'cluster_avg'")

    # if cond_parameters is a list of str, assume those are cond parameter names
    if isinstance(cond_parameters[0], str):
        cond_parameter_idx = [parameter_names.index(p) for p in cond_parameters]
        cond_parameter_idx.sort()
    # else if it's a list of int, assume those are cond parameter indices
    elif isinstance(cond_parameters[0], int):
        cond_parameter_idx = sorted(cond_parameters)

    n_cond_parameters = len(cond_parameter_idx)
    cond_parameter_names = [parameter_names[i] for i in cond_parameter_idx]

    # Convert label array to dict of indices
    clusters = {}
    for nc in range(n_clusters):
        cluster = np.where(labels == nc)[0]
        clusters[nc] = cluster

    # Calculate the thresholds separating bins for each parameter
    # thresholds is of shape (n_bins, n_parameters)
    thresholds = np.quantile(parameters, [b / n_bins for b in range(1, n_bins)], axis=0)
    percentiles = np.arange(1, 100)

    # NaN-filled arrays of L1 distances and alpha-quantile L1 distances
    param_interact = np.empty((n_cond_parameters, n_parameters - 1, n_clusters, n_bins), dtype="float64")
    param_interact[:] = np.nan
    boot_interact = np.empty((n_cond_parameters, n_parameters - 1, n_clusters, n_bins), dtype="float64")
    boot_interact[:] = np.nan

    # Loop through each conditional parameter and calculate
    if progress:
        iterator = tqdm(cond_parameter_idx, desc="Performing DGSA")
    else:
        iterator = cond_parameter_idx

    for i, cond_idx in enumerate(iterator):
        param_interact[i] = interact_distance(cond_idx, parameters, clusters, thresholds, percentiles)
        boot_interact[i] = interact_boot_distance(
            cond_idx, parameters, clusters, thresholds, percentiles, n_boots=n_boots, alpha=quantile, progress=progress
        )

    # Calculate the normalized distances (d/d_95)
    normalized_interactions = param_interact / boot_interact

    # Average over each bin
    sensitivity_per_class = np.nanmean(normalized_interactions, axis=3)

    # Average over each cluster
    sensitivity_per_bin = np.nanmean(normalized_interactions, axis=2)

    # Average over each cluster and bin
    sensitivity_over_class = np.nanmean(sensitivity_per_class, axis=2)

    ### Choose how to output the results
    # Output mean sensitvitiy as df, with indices as interactions (e.g., 'x | y')
    if output == "mean":
        df = pd.DataFrame(columns=["sensitivity"])

        for i, cond_param in enumerate(cond_parameter_names):
            main_params = [x for x in parameter_names if x != cond_param]

            for j, main_param in enumerate(main_params):
                interact_name = main_param + " | " + cond_param
                df.loc[interact_name, "sensitivity"] = sensitivity_over_class[i, j]
        df.sort_values(by="sensitivity", ascending=False, inplace=True)

    # Output mean sensitivity in array form, with rows as cond params and
    # cols as main params
    elif output == "mat":
        df = pd.DataFrame(columns=parameter_names)

        for i, cond_param in enumerate(cond_parameter_names):
            main_params = [x for x in parameter_names if x != cond_param]

            for j, main_param in enumerate(main_params):
                df.loc[cond_param, main_param] = sensitivity_over_class[i, j]

    # Output sensitivity of each cluster, indices as interactions, cols as clusters
    elif output == "cluster_avg":
        if cluster_names is None:
            cluster_names = ["Cluster {}".format(i) for i in range(n_clusters)]
        df = pd.DataFrame(columns=cluster_names)

        for i, cond_param in enumerate(cond_parameter_names):
            main_params = [x for x in parameter_names if x != cond_param]

            for j, main_param in enumerate(main_params):
                interact_name = main_param + " | " + cond_param
                df.loc[interact_name, :] = sensitivity_per_class[i, j, :]

    # Output sensitivity of each cluster/bin combination with indices as
    # interactions (e.g., 'x | y') and cols as a cluster/bin combination
    elif output == "bin_avg":
        bin_names = ["Bin {}".format(i) for i in range(n_bins)]
        df = pd.DataFrame(columns=bin_names)

        for i, cond_param in enumerate(cond_parameter_names):
            main_params = [x for x in parameter_names if x != cond_param]

            for j, main_param in enumerate(main_params):
                interact_name = main_param + " | " + cond_param
                df.loc[interact_name, :] = sensitivity_per_bin[i, j, :]

    # Output sensitivity of each cluster/bin combination with indices as
    # interactions (e.g., 'x | y') and cols as a cluster/bin combination
    elif output == "indiv":
        if cluster_names is None:
            cluster_names = ["Cluster {}".format(i) for i in range(n_clusters)]
        bin_names = ["Bin {}".format(i) for i in range(n_bins)]
        columns = pd.MultiIndex.from_product([cluster_names, bin_names])
        df = pd.DataFrame(columns=columns)
        idx = pd.IndexSlice

        for i, cond_param in enumerate(cond_parameter_names):
            main_params = [x for x in parameter_names if x != cond_param]

            for j, main_param in enumerate(main_params):
                interact_name = main_param + " | " + cond_param

                for k, cluster in enumerate(cluster_names):
                    df.loc[interact_name, idx[cluster, :]] = normalized_interactions[i, j, k, :]

    return df
