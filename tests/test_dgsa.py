import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from pyDGSA.cluster import KMedoids
from pyDGSA.dgsa import cluster_cdf_distance, dgsa, dgsa_interactions

# Generate analytic parameters and calculate labels
@pytest.fixture
def analytic_data():
    # Load parameters
    parameters = np.load('tests/data/analytic_params.npy')
    # Calculate responses
    responses = np.empty((parameters.shape[0], 4), dtype='float64')
    responses[:, 0] = parameters[:, 4]
    responses[:, 1] = np.abs(parameters[:, 2] * parameters[:, 3] - 1)
    responses[:, 2] = np.sqrt(np.minimum(parameters[:, 2], parameters[:, 3]))
    responses[:, 3] = np.sqrt(parameters[:, 4])

    # Calculate pairwise Euclidean distances between responses
    distances = pdist(responses, metric='euclidean')
    distances = squareform(distances)

    # Cluster the distances using KMedoids
    n_clusters = 3
    clusterer = KMedoids(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
    labels, _ = clusterer.fit_predict(distances)

    return parameters, labels

def test_cluster_cdf_distance():
    prior_cdf = np.linspace(0, 1, 99).reshape(99, 1)
    cluster_params = np.random.uniform(0, 1, (40, 1))
    distances = cluster_cdf_distance(prior_cdf, cluster_params)
    assert distances.shape == (1,)
    assert distances[0] > 0  # Distance should be non-zero for random inputs

def test_dgsa_analytic(analytic_data):
    parameters, labels = analytic_data

    # Run DGSA
    result = dgsa(parameters, labels, parameter_names=['v', 'w', 'x', 'y', 'z'],
                  n_boots=3000, progress=False)

    # Validate structure of output
    assert result.shape[0] == 5
    assert 'sensitivity' in result.columns

    # Validate sensitivities
    assert result.loc['z', 'sensitivity'] > 1  # High influence
    assert result.loc['v', 'sensitivity'] < 1  # Low influence
    assert result.loc['w', 'sensitivity'] < 1  # Low influence

def test_dgsa_interactions_analytic(analytic_data):
    parameters, labels = analytic_data

    # Run DGSA Interactions
    result = dgsa_interactions(parameters, labels, cond_parameters=['x', 'v'],
                               n_bins=4, output='mean', progress=False,
                               parameter_names=['w', 'v', 'x', 'y', 'z'])

    # Validate output structure
    assert result.shape[0] == 8  # Interactions with the conditional parameter
    assert 'sensitivity' in result.columns

    correct_rows = ['v | x', 'w | x', 'y | x', 'z | x',
                    'w | v', 'x | v', 'y | v', 'z | v']
    assert all([row in result.index for row in correct_rows])

    # Check that interaction 'w | v' does not have meaningful sensitivity
    # This may fail due to random variations, though it's unlikely
    assert result.loc['w | v', 'sensitivity'] < 1
