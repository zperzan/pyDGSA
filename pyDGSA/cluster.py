# -*- coding: utf-8 -*-

"""
pyDGSA sub-package that provides clustering using KMedoids.

Credit for this sub-package goes to:
David Zhen Yin, Aug 2019

Updated:
Zach Perzan, June 2020
Zach Perzan, Mar 2023
Zach Perzan, July 2024
"""

import numpy as np


def _get_init_centers(n_clusters, n_samples):
    """Calculate initial cluster centers.

    Parameters
    ----------
    n_clusters : int
        Number of clusters
    n_samples : int
        Number of samples

    Returns
    -------
    init_ids : list
        The initial indices of the center of each cluster. Each index corresponds
        to an individual sample
    """
    init_ids = []
    while len(init_ids) < n_clusters:
        rand_center = np.random.randint(0, n_samples)
        if rand_center not in init_ids:
            init_ids.append(rand_center)
    return init_ids


def _get_cost(dist_meds, current_medoids):
    """Calculate the total cost and cost of each cluster

    Parameters
    ----------
    dist_meds : ndarray
        Paired distances between all data points and each medoid
    current_medoids : ndarray
        The locations of each current medioid

    Returns
    -------
    total cost : ndarray
        the total cost of current distance between all medoids
    """
    costs = np.zeros(len(current_medoids))
    dis_min = np.min(dist_meds, axis=0)
    for i in range(len(current_medoids)):
        clst_mem_ids = np.where(dist_meds[i] == dis_min)[0]
        costs[i] = np.sum(dist_meds[i][clst_mem_ids])
    return np.sum(costs)


def _kmedoids_run(x, n_clusters, max_iter=3000, tol=1e-4):
    """Perform k-medoids clustering for a given dataset

    Parameters
    ----------
    x : ndarray
        The squareform distance array for all samples, of shape (#samples, #samples)
    n_clusters : ndarray
        Number of clusters
    max_iter : int
        Maximum number of iterations to perform. Optional, defaults to 3000
    tol : float
        The tolerance at which to stop the iterations, in percentage. For example,
        if tol=0.01, then the iterations will stop when the cost function decrease
        is less than 1%. Optional, defaults to 1e-4.

    Returns
    -------
    current_medoids : ndarray
        Locations of the final medoids, of shape (# clusters, # samples)
    clsts_membr_ids : ndarray
        The members of each cluster
    costs_iters : ndarray
        The cost function evaluation of each iteration
    """
    n_samples = x.shape[0]

    # Initialize the medoids
    current_medoids = np.asarray(_get_init_centers(n_clusters, n_samples))

    # Calculate the total cost of the initial medoids
    costs_iters = []
    dist_meds = x[current_medoids]
    tot_cos = _get_cost(dist_meds, current_medoids)
    costs_iters.append(tot_cos)
    cc = 0

    for i in range(max_iter):
        dist_meds = x[current_medoids]
        # Associate  each data point to the closest medoid
        # And calculate the total cost
        tot_cos = _get_cost(dist_meds, current_medoids)

        # Get new medoids
        new_medoids = []
        while len(new_medoids) == 0:
            for j in range(n_clusters):
                o = np.random.choice(n_samples)
                if o not in current_medoids and o not in new_medoids:
                    new_medoids.append(o)
        new_medoids = np.asarray(new_medoids).astype(int)
        dist_meds_ = x[new_medoids]

        tot_cos_ = _get_cost(dist_meds_, new_medoids)
        # Swap new_medoids with the current medoid if cost decreases
        if (tot_cos_ - tot_cos) < 0:
            current_medoids = new_medoids
            costs_iters.append(tot_cos_)
            cc += 1
            if abs(costs_iters[cc] / costs_iters[cc - 1] - 1) < tol:
                # Associated  data points to the final calculated medoids (reached by tolerance)
                clsts_membr_ids = []
                dis_min = np.min(dist_meds, axis=0)
                for k in range(n_clusters):
                    clst_mem_ids = np.where(dist_meds[k] == dis_min)[0]
                    clsts_membr_ids.append(clst_mem_ids)

                return current_medoids, clsts_membr_ids, costs_iters

    costs_iters = np.asarray(costs_iters)
    # Associated  data points to the final calculated medoids (reached by maximum iters)
    clsts_membr_ids = []
    dist_meds = x[current_medoids]
    dis_min = np.min(dist_meds, axis=0)
    for k in range(n_clusters):
        clst_mem_ids = np.where(dist_meds[k] == dis_min)[0]
        clsts_membr_ids.append(clst_mem_ids)

    return current_medoids, clsts_membr_ids, costs_iters


class KMedoids(object):
    """Main class for KMedoids clustering

    Parameters
    ----------
    n_clusters : int
        Number of clusters
    max_iter : int
        Maximum number of iterations to perform. Optional, defaults to 10000
    tol : float
        The tolerance at which to stop the iterations, in percentage. For example,
        if tol=0.01, then the iterations will stop when the cost function decrease
        is less than 1%. Optional, defaults to 1e-4.

    Attributes
    ----------
    costs_itr : ndarray
        The cost function evaluation of each iteration
    cluster_centers_ : ndarray
        Locations of the final medoids, of shape (# clusters, # samples)
    n_clusters : int
        Number of clusters
    max_iter : int
        Maximum number of iterations to perform. Optional, defaults to 10000
    tol : float
        The tolerance at which to stop the iterations, in percentage. For example,
        if tol=0.01, then the iterations will stop when the cost function decrease
        is less than 1%. Optional, defaults to 1e-4.

    Methods
    -------
    fit_predict
        Perform k-medoids clustering for a given dataset

    Examples
    --------
    >>> model = KMedoids(n_clusters=3)
    >>> medoids, cluster_id = model.fit_predict(x)
    """

    def __init__(self, n_clusters, max_iter=10000, tol=0.005, random_state=None):
        self.costs_itr = None
        self.cluster_centers_ = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

        if random_state is not None:
            np.random.seed(random_state)

    def fit_predict(self, x):
        """Run the main k-medoids function

        Parameters
        ----------
        x : ndarray
            The squareform distance array for all samples, of shape (#samples, #samples)

        Returns
        -------
        labels : ndarray
            Cluster membership for each of the samples
        cluster_centers_ : ndarray
            Indices corresponding to the medoids of each cluster
        """
        self.cluster_centers_, clst_membr_ids, self.costs_itr = _ = _kmedoids_run(
            x, self.n_clusters, self.max_iter, self.tol
        )

        # Convert clst_membr_ids to labels
        labels = np.ones(x.shape[0], dtype="int32")
        for i in range(self.n_clusters):
            labels[clst_membr_ids[i]] = i

        return labels, self.cluster_centers_
