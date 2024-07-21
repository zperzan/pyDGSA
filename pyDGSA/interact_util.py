# -*- coding: utf-8 -*-

"""
pyDGSA sub-package that provides helper functions used in the 
pyDGSA.dgsa.dgsa_interactions submodule

This file includes:
interact_boot_distances
interact_distance
"""

import numpy as np
from tqdm.notebook import tqdm


def interact_distance(cond_idx, parameters, clusters, thresholds, percentiles):
    """For a single conditional parameter, calculate the L1-norm distance between 
    the class-conditional distributions of all other parameters and the class-
    conditional distributions additionally conditioned on another parameter. 
    
    params:
        cond_idx [int]: index of the conditional parameter within the columns 
                of the parameters array
        parameters [array]: array of shape (n_simulations, n_parameters) 
                containing parameters used to generate each simulation
        clusters [dict(array)]: dict of len(n_clusters). Values are the indices 
                within each cluster, corresponding to the rows of parameters
        thresholds [array]: array of shape (n_bins-1, n_parameters) containing the 
                thresholds at which to separate each conditional parameter
        percentiles [array]: array of percentiles at which to evaluate cdf
    
    returns:
        cdf_distances [array]: array of shape (n_parameters-1, n_clusters, 
                n_bins) containing the L1-norm distances for each 
                parameter/cluster/bin combination
    """

    # Get number of params, bins and clusters
    n_parameters = parameters.shape[1]
    n_bins = thresholds.shape[0] + 1
    n_clusters = len(clusters)

    cdf_distances = np.zeros((n_parameters, n_clusters, n_bins), dtype='float64')

    # Loop through cluster and bin
    for nc in range(n_clusters):
        # Indices comprising this cluster
        c_idx = clusters[nc] 
        
        # Priors for all parameters, ie F(p|C)
        q_prior = np.percentile(parameters[c_idx, :], percentiles, axis=0) 

        for nb in range(n_bins): 
            # For each bin, first calc b_mask -- mask of params within each bin    
            # The first bin
            if nb == 0: 
                threshold = thresholds[nb, cond_idx]
                b_mask = parameters[:, cond_idx] <= threshold
            # The last bin
            elif nb == (n_bins - 1): 
                threshold = thresholds[-1, cond_idx]
                b_mask = parameters[:, cond_idx] > threshold
            # Middle bins
            else: 
                low_thresh = thresholds[nb - 1, cond_idx]
                high_thresh = thresholds[nb, cond_idx]
                b_mask = (parameters[:, cond_idx] > low_thresh) & \
                         (parameters[:, cond_idx] <= high_thresh)

            # Indices comprising this bin
            b_idx = np.argwhere(b_mask) 
            
            # Indices within this cluster AND bin
            bc_idx = np.intersect1d(c_idx, b_idx.flatten(), assume_unique=True) 

            # If no params within a cluster/bin, distance = nan
            if len(bc_idx) == 0:
                cdf_distances[:, nc, nb] = np.nan
            else:
                q_inter = np.percentile(parameters[bc_idx, :], percentiles, axis=0) 
                cdf_distances[:, nc, nb] = np.sum(abs(q_inter - q_prior), axis=0)

    # Delete row of zeros (parameter conditioned on itself) from array
    cdf_distances = np.delete(cdf_distances, cond_idx, axis=0)
                    
    return cdf_distances


def interact_boot_distance(cond_idx, parameters, clusters, thresholds, 
                           percentiles, n_boots=3000, alpha=0.95,
                           progress=True):
    """For a single parameter, performs a resampling procedure that calculates
    the alpha-quantile of the L1-norm for each parameter given the conditional 
    parameter, each cluster and each bin.
    
    params:
        cond_idx [int]: index of the conditional parameter within the columns 
                of the parameters array
        parameters [array]: array of shape (n_simulations, n_parameters) 
                containing parameters used to generate each simulation
        clusters [dict(array)]: dict of len(n_clusters). Values are the indices 
                within each cluster, corresponding to the rows of parameters
        thresholds [array]: array of shape (n_bins-1, n_parameters) containing the 
                thresholds at which to separate each conditional parameter
        percentiles [array]: array of percentiles at which to evaluate cdf
        n_boots [int]: number of iterations to perform resampling. Default: 3000
        alpha [float]: alpha-quantile at which to evaluate L1-norm. Can be 
                either [0, 1] (quantile) or [0, 100] (percentile)
        progress [bool]: whether to display tqdm progress bar during calculation.
                Default is True
    
    returns:
        boot_distances [array]: array of shape (n_parameters-1, n_clusters, 
                n_bins) containing the alpha-quantile L1-norm distances for 
                each parameter/cluster/bin combination
    """

    # Check alpha input and scale as needed
    if (alpha >= 0) & (alpha <= 1):
        pass
    elif alpha < 0:
        raise ValueError("alpha-percentile must be greater than zero")
    elif alpha > 100:
        raise ValueError("alpha-percentile must be between 0 and 100 inclusive")
    else:
        alpha = alpha / 100
    
    # Get number of params, bins and clusters
    n_parameters = parameters.shape[1]
    n_bins = thresholds.shape[0] + 1
    n_clusters = len(clusters)

    boot_distances = np.zeros((n_parameters, n_clusters, n_bins), dtype='float64')

    # Loop through each parameter, cluster and bin
    if progress:
        iterator = tqdm(range(n_clusters), desc='Resampling parameter %d' % cond_idx, leave=False)
    else:
        iterator = range(n_clusters)
    for nc in iterator:
        # Indices comprising this cluster
        c_idx = clusters[nc] 
        
        # Priors for all parameters, ie F(p|C)
        q_prior = np.percentile(parameters[c_idx, :], percentiles, axis=0) 

        for nb in range(n_bins): 
            # The first bin
            if nb == 0: 
                threshold = thresholds[nb, cond_idx]
                b_mask = parameters[:, cond_idx] <= threshold
            # The last bin
            elif nb == (n_bins - 1): 
                threshold = thresholds[-1, cond_idx]
                b_mask = parameters[:, cond_idx] > threshold
            # Middle bins
            else: 
                low_thresh = thresholds[nb - 1, cond_idx]
                high_thresh = thresholds[nb, cond_idx]
                b_mask = (parameters[:, cond_idx] > low_thresh) & (parameters[:, cond_idx] <= high_thresh)

            b_idx = np.argwhere(b_mask)
            bc_idx = np.intersect1d(c_idx, b_idx.flatten(), assume_unique=True)
            
            # If fewer than 4 values in this bin, set to nan
            if len(bc_idx) <= 3:
                boot_distances[:, nc, nb] = np.nan
            else:
                boot_dist = np.zeros((n_boots, n_parameters))
                for n_boot in range(n_boots):
                    boot_idx = np.random.choice(c_idx, len(bc_idx), replace=False)
                    boot_inter = np.percentile(parameters[boot_idx, :], percentiles, axis=0) # cdf of cond_idx conditioned to n_p in bin nb
                    boot_dist[n_boot] = np.sum(abs(boot_inter - q_prior), axis=0)
                
                boot_dist[:, cond_idx] = np.nan
                boot_distances[:, nc, nb] = np.quantile(boot_dist, alpha, axis=0)

    # Delete row of zeros (parameter conditioned on itself) from array
    boot_distances = np.delete(boot_distances, cond_idx, axis=0)

    return boot_distances


