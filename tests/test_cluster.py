import numpy as np
import pytest
from pyDGSA.cluster import _get_init_centers, _get_cost, _kmedoids_run, KMedoids


def test_get_init_centers():
    centers = _get_init_centers(3, 10)
    assert len(centers) == 3
    assert len(set(centers)) == 3
    assert all(0 <= center < 10 for center in centers)

def test_kmedoids_run():
    dist_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    medoids, clusters, costs = _kmedoids_run(dist_matrix, n_clusters=2, max_iter=10)
    assert len(medoids) == 2
    assert len(clusters) == 2
    assert all(isinstance(cluster, np.ndarray) for cluster in clusters)

def test_kmedoids_class():
    model = KMedoids(n_clusters=2, max_iter=10, tol=0.1)
    dist_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    labels, medoids = model.fit_predict(dist_matrix)
    assert len(medoids) == 2
    assert len(labels) == dist_matrix.shape[0]
