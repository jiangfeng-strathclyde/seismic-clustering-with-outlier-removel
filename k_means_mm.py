import copy
import warnings
from utils import *
# from model import *
import torch.optim as optim
import csv
import numpy as np
import scipy.sparse as sp


from sklearn.cluster.k_means_ import KMeans
from sklearn.cluster.k_means_ import _labels_inertia, _check_sample_weight
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.metrics import normalized_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import _num_samples
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed

def _check_normalize_sample_weight(sample_weight, X):
    """Set sample_weight if None, and check for correct dtype"""

    sample_weight_was_none = sample_weight is None

    sample_weight = _check_sample_weight(sample_weight, X)

    if not sample_weight_was_none:
        # normalize the weights to sum up to n_samples
        # an array of 1 (i.e. samples_weight is None) is already normalized
        n_samples = len(sample_weight)
        scale = n_samples / sample_weight.sum()
        sample_weight *= scale
    return sample_weight

def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if len(centers) != n_centers:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, n_centers))
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial centers %s "
            "does not match the number of features of the data %s."
            % (centers.shape[1], X.shape[1]))

def _k_means_minus_minus(
    X,
    sample_weight,
    n_clusters,
    prop_outliers,
    max_iter=300,
    init="k-means++",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
    precompute_distances=True,
):
    """A single run of k-means, assumes preparation completed prior.
    Parameters
    ----------
    X : array-like of floats, shape (n_samples, n_features)
        The observations to cluster.
    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.
    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.
    prop_outliers : float
        What proportion of the training dataset X to treat as outliers, and
        to exclude in each iteration of Lloyd's algorithm.
    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.
    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.
        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.
    tol : float, optional
        The relative increment in the results before declaring convergence.
    verbose : boolean, optional
        Verbosity mode
    x_squared_norms : array
        Precomputed x_squared_norms.
    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.
    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.
    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).
    n_iter : int
        Number of iterations run.
    """

    n_outliers = int(X.shape[0] * prop_outliers)
    random_state = check_random_state(random_state)

    sample_weight = _check_normalize_sample_weight(sample_weight, X)

    best_labels, best_inertia, best_centers = None, None, None
    # init
    centers = _init_centroids(
        X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms
    )
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()

        # labels assignment is also called the E-step of EM
        labels, inertia = _labels_inertia(
            X,
            sample_weight,
            x_squared_norms,
            centers,
            precompute_distances=precompute_distances,
            distances=distances,
        )

        # the "minus-minus" modification step - filter out n_outliers # of
        # datapoints that are farthest from their assigned cluster centers
        X_subset, sample_weight_subset, labels_subset, distances_subset = (
            X,
            sample_weight,
            labels,
            distances,
        )

        outlier_indices = 1
        if n_outliers > 0:
            outlier_indices = np.argpartition(distances, -n_outliers)[
                -n_outliers:
            ]  # ~20x faster than np.argsort()

            X_subset, sample_weight_subset, labels_subset, distances_subset = (
                np.delete(X, outlier_indices, axis=0),
                np.delete(sample_weight, outlier_indices, axis=0),
                np.delete(labels, outlier_indices, axis=0),
                np.delete(distances, outlier_indices, axis=0),
            )

            # indices_to_refit = np.argsort(distances) < (X.shape[0] - n_outliers)
        # X_subset, sample_weight_subset = X[indices_to_refit], sample_weight[indices_to_refit]

        # computation of the means is also called the M-step of EM
        if sp.issparse(X):

            centers = centers_sparse(
                X_subset, sample_weight_subset, labels_subset, n_clusters, distances_subset
            )
        else:
            centers = centers_dense(
                X_subset, sample_weight_subset, labels_subset, n_clusters, distances_subset
            )

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = np.linalg.norm((centers_old - centers)) ** 2
        if center_shift_total <= tol:
            if verbose:
                print(
                    "Converged at iteration %d: "
                    "center shift %e within tolerance %e" % (i, center_shift_total, tol)
                )
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = _labels_inertia(
            X,
            sample_weight,
            x_squared_norms,
            best_centers,
            precompute_distances=precompute_distances,
            distances=distances,
        )

    return best_labels, best_inertia, best_centers, outlier_indices

def centers_sparse(X, sample_weight, labels, n_clusters, distances):
    """
    M step of the K-means EM algorithm
    Computation of cluster centers / means.
    :param X: scipy.sparse.csr_matrix, shape (n_samples, n_features)
    :param sample_weight: array-like, shape (n_samples,)
        The weights for each observation in X.
    :param labels: array of integers, shape (n_samples)
        Current label assignment
    :param n_clusters: int
        Number of desired clusters
    :param distances: array-like, shape (n_samples)
        Distance to closest cluster for each sample.
    :return: centers, array, shape (n_clusters, n_features)
        The resulting centers
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    data = X.data
    indices = X.indices
    indptr = X.indptr

    dtype = X.dtype
    centers = np.zeros((n_clusters, n_features), dtype=dtype)
    weight_in_cluster = np.zeros((n_clusters,), dtype=dtype)
    for i in range(n_samples):
        c = labels[i]
        weight_in_cluster[c] += sample_weight[i]
    empty_clusters = np.where(weight_in_cluster == 0)[0]
    n_empty_clusters = empty_clusters.shape[0]

    # maybe also relocate small clusters?

    if n_empty_clusters > 0:
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1][:n_empty_clusters]
        assign_rows_csr(X, far_from_centers, empty_clusters, centers)

        for i in range(n_empty_clusters):
            weight_in_cluster[empty_clusters[i]] = 1

    for i in range(labels.shape[0]):
        curr_label = labels[i]
        for ind in range(indptr[i], indptr[i + 1]):
            j = indices[ind]
            centers[curr_label, j] += data[ind] * sample_weight[i]

    centers /= weight_in_cluster[:, np.newaxis]

    return centers

def centers_dense(X, sample_weight, labels, n_clusters, distances):
    """
    M step of the K-means EM algorithm
    Computation of cluster centers / means.
    :param X: array-like, shape (n_samples, n_features)
    :param sample_weight: array-like, shape (n_samples,)
        The weights for each observation in X.
    :param labels: array of integers, shape (n_samples)
        Current label assignment
    :param n_clusters: int
        Number of desired clusters
    :param distances: array-like, shape (n_samples)
        Distance to closest cluster for each sample.
    :return: centers : array, shape (n_clusters, n_features)
        The resulting centers
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    dtype = X.dtype
    centers = np.zeros((n_clusters, n_features), dtype=dtype)
    weight_in_cluster = np.zeros((n_clusters,), dtype=dtype)

    for i in range(n_samples):
        c = labels[i]
        weight_in_cluster[c] += sample_weight[i]
    empty_clusters = np.where(weight_in_cluster == 0)[0]
    # maybe also relocate small clusters?

    if distances is not None and len(empty_clusters):
        # find points to reassign empty clusters to
        far_from_centers = distances.argsort()[::-1]

        for i, cluster_id in enumerate(empty_clusters):
            far_index = far_from_centers[i]
            new_center = X[far_index] * sample_weight[far_index]
            centers[cluster_id] = new_center
            weight_in_cluster[cluster_id] = sample_weight[far_index]

    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j] * sample_weight[i]

    centers /= weight_in_cluster[:, np.newaxis]

    return centers

if __name__=="__main__":

    path = ## your data path
    x, y = load_time_seties_data(path) 
    y_onedim = np.argmax(y, axis=1)
    filtered_array, f, x_stft_normalized_spectrogram = bandpass_stft_filter(x, low_frequency=1, upper_frequency=20, fs=250)
    x_stft_normalized_spectrogram = np.nan_to_num(x_stft_normalized_spectrogram)
    No_samples = 
    No_channels = 
    No_frequency_bin = 
    No_time_bin = 
    X = np.reshape(x_stft_normalized_spectrogram,[No_samples,No_channels*No_frequency_bin*No_time_bin])
    sample_weight = np.ones([1375])
    acc_value = np.array([])
    prop_outliers_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
    num_run = 20
    for i_prop in range(len(prop_outliers_list)):
        acc_value = np.array([])
        for i_run in range(num_run):
            best_labels, best_inertia, best_centers,_ = _k_means_minus_minus(X, sample_weight = sample_weight, n_clusters=4,prop_outliers=prop_outliers_list[i_prop], max_iter=300, init="k-means++", verbose=False, x_squared_norms=None, tol=1e-4, precompute_distances=True)
            acc_value = np.concatenate([acc_value, np.array([np.round(acc(y_onedim, best_labels), 5)])])
            nmi = np.round(normalized_mutual_info_score(y_onedim, best_labels), 5)
            
            # Construct the file path
            acc_value_list = [acc_value]
            file_path = 'k_means_mm_accwithprop_outliers_{}.csv'.format(str(prop_outliers_list[i_prop]))

            # Save data to CSV
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([acc_value_list])  # Pass acc_value_list as the iterable
            acc_value_list = []
            print('k-means_MM acc = %f', np.round(acc(y_onedim, best_labels), 5))

            # Construct the file path
            nmi_list = [nmi]
            file_path = 'k_means_mm_nmiwithprop_outliers_{}.csv'.format(str(prop_outliers_list[i_prop]))

            # Save data to CSV
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([nmi_list])  # Pass acc_value_list as the iterable
            nmi_list = []
            print('k-means_MM nmi = %f', nmi)