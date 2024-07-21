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
        n_clusters: number of clusters
        n_samples: number of samples

    Returns
    -------
        init_ids: the initial cluster centers' ids"""

    init_ids = []
    while len(init_ids) < n_clusters:
        _ = np.random.randint(0,n_samples)
        if not _ in init_ids:
            init_ids.append(_)
    return init_ids


def _get_cost(dist_meds, currentMedoids):
    """Calculate the total cost and cost of each cluster

    Parameters
    ----------
        currentMedoids: the current Medoids
        dist_meds: paired distances between all data points and each Medoid

    Returns
    -------
        total cost"""

    costs = np.zeros(len(currentMedoids))
    dis_min = np.min(dist_meds,axis=0)
    for i in range(len(currentMedoids)):
        clst_mem_ids = np.where(dist_meds[i] == dis_min)[0]
        costs[i] = np.sum(dist_meds[i][clst_mem_ids])
    return np.sum(costs)


def _kmedoids_run(X, n_clusters, max_iter=3000, tol=1e-4):
    """Main function for runing the k-medoids clustering
    Parameters
    ----------
        X: the squareform distance array for all samples, (#samples, #samples)
        n_cluster: number of clusters
        max_iter: maximum number of clusters
        tol: the tolerance to stop the iterations, in percentage;
                i.e.: if tol=0.01, it means if the cost function decrease is less than 1%, the iteraction will stop.

    Returns
    -------
        currentMedoids: the final medoids
        clsts_membr_ids: the members of each cluster
        costs_iters: the costs of each iteration"""
    
    n_samples = X.shape[0]

    # Initialize the medoids
    currentMedoids = np.asarray(_get_init_centers(n_clusters,n_samples))
    
    # Calculate the total cost of the initial medoids
    costs_iters=[]
    dist_meds = X[currentMedoids]
    tot_cos = _get_cost(dist_meds,currentMedoids)
    costs_iters.append(tot_cos)
    cc = 0
    
    for i in range(max_iter):
        dist_meds = X[currentMedoids]
        # Associate  each data point to the closest medoid
        # And calculate the total cost
        tot_cos = _get_cost(dist_meds, currentMedoids)

        # Get new mediods
        newMedoids = []
        while len(newMedoids) == 0:
            for j in range(n_clusters):
                o = np.random.choice(n_samples)
                if o not in currentMedoids and o not in newMedoids:
                    newMedoids.append(o)
        newMedoids = np.asarray(newMedoids).astype(int)
        dist_meds_ = X[newMedoids]

        tot_cos_ = _get_cost(dist_meds_, newMedoids)
        # Swap newmediods with the current mediod if cost decreases
        if (tot_cos_ - tot_cos) < 0:
            currentMedoids = newMedoids
            costs_iters.append(tot_cos_)
            cc += 1
            if abs(costs_iters[cc]/costs_iters[cc-1]-1) < tol:
                # Associated  data points to the final calucated medoids (reached by tolerance)
                clsts_membr_ids = []
                dis_min = np.min(dist_meds, axis=0)
                for k in range(n_clusters):
                    clst_mem_ids = np.where(dist_meds[k] == dis_min)[0]
                    clsts_membr_ids.append(clst_mem_ids)

                return currentMedoids, clsts_membr_ids, costs_iters
            
    costs_iters = np.asarray(costs_iters)
    # Associated  data points to the final calculated medoids (reached by maximum iters)
    clsts_membr_ids = []
    dist_meds = X[currentMedoids]
    dis_min = np.min(dist_meds, axis=0)
    for k in range(n_clusters):
        clst_mem_ids = np.where(dist_meds[k] == dis_min)[0]
        clsts_membr_ids.append(clst_mem_ids)

    return currentMedoids, clsts_membr_ids, costs_iters


class KMedoids(object):
    """Main API of KMedoids Clustering
    Parameters
    --------
        X: the input ndarray data for k-medoids clustering, (#samples, #features)
        n_clusters: number of clusters
        max_iter: maximum number of iterations
        tol:  the tolerance to stop the iterations, in percentage; 
                    i.e.: if tol=0.01, it means if the cost function decrease is less than 1%, the iterations will stop.
    Attributes
    --------
        Medoids:  cluster Medoids id
        costs_itr   :  array of costs for each effective iterations
        clst_membr_ids   :  each cluster members' sample-ids in the input ndarray X 
    
    Methods
    -------
        model = KMedoids(n_clusters = #, max_iter=#<opertional>, tol==#<opertional>)
        Medoids, cluster_id = model.fit(X, plotit=True/False): fit the model, 
                                                    it returns the center medoids id in the ndarray X, 
                                                    and  each cluster members' sample-ids in X."""
    def __init__(self, n_clusters, max_iter=10000, tol=0.005):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit_predict(self, X):
        """Run the main k-medoids function"""
        self.cluster_centers_, clst_membr_ids, self.costs_itr = _=_kmedoids_run(X,
                                                                                self.n_clusters,
                                                                                self.max_iter,
                                                                                self.tol)
        
        # Convert clst_membr_ids to labels
        labels = np.ones(X.shape[0], dtype='int32')
        for i in range(self.n_clusters):
            labels[clst_membr_ids[i]] = i

        return labels, self.cluster_centers_
