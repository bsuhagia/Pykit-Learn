"""This module provides clustering utility functions.
    Author: Bhavesh
"""
from sklearn import cluster
from sklearn.mixture import GMM, DPGMM

def train_gmm(X, n_components=3, covariance_type='diag', random_state=None,
              thresh=None, tol=0.001, min_covar=0.001, n_iter=100, n_init=1,
              params='wmc', init_params='wmc'):
    """Variational Inference for the Infinite Gaussian Mixture Model.

    DPGMM stands for Dirichlet Process Gaussian Mixture Model, and it
    is an infinite mixture model with the Dirichlet Process as a prior
    distribution on the number of clusters. In practice the
    approximate inference algorithm uses a truncated distribution with
    a fixed maximum number of components, but almost always the number
    of components actually used depends on the data.

    Stick-breaking Representation of a Gaussian mixture model
    probability distribution. This class allows for easy and efficient
    inference of an approximate posterior distribution over the
    parameters of a Gaussian mixture model with a variable number of
    components (smaller than the truncation parameter n_components).

    Initialization is with normally-distributed means and identity
    covariance, for proper convergence.

    Parameters
    ----------
    n_components: int, optional
        Number of mixture components. Defaults to 1.

    covariance_type: string, optional
        String describing the type of covariance parameters to
        use.  Must be one of 'spherical', 'tied', 'diag', 'full'.
        Defaults to 'diag'.

    alpha: float, optional
        Real number representing the concentration parameter of
        the dirichlet process. Intuitively, the Dirichlet Process
        is as likely to start a new cluster for a point as it is
        to add that point to a cluster with alpha elements. A
        higher alpha means more clusters, as the expected number
        of clusters is ``alpha*log(N)``. Defaults to 1.

    thresh : float, optional
        Convergence threshold.
    n_iter : int, optional
        Maximum number of iterations to perform before convergence.
    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.
    init_params : string, optional
        Controls which parameters are updated in the initialization
        process.  Can contain any combination of 'w' for weights,
        'm' for means, and 'c' for covars.  Defaults to 'wmc'.

    Attributes
    ----------
    covariance_type : string
        String describing the type of covariance parameters used by
        the DP-GMM.  Must be one of 'spherical', 'tied', 'diag', 'full'.

    n_components : int
        Number of mixture components.

    `weights_` : array, shape (`n_components`,)
        Mixing weights for each mixture component.

    `means_` : array, shape (`n_components`, `n_features`)
        Mean parameters for each mixture component.

    `precs_` : array
        Precision (inverse covariance) parameters for each mixture
        component.  The shape depends on `covariance_type`::

            (`n_components`, 'n_features')                if 'spherical',
            (`n_features`, `n_features`)                  if 'tied',
            (`n_components`, `n_features`)                if 'diag',
            (`n_components`, `n_features`, `n_features`)  if 'full'

    `converged_` : bool
        True when convergence was reached in fit(), False otherwise.

    See Also
    --------
    GMM : Finite Gaussian mixture model fit with EM

    VBGMM : Finite Gaussian mixture model fit with a variational
        algorithm, better for situations where there might be too little
        data to get a good estimate of the covariance matrix.
    """

    model = GMM(n_components=n_components,
                covariance_type=covariance_type,
                random_state=random_state,
                thresh=thresh,
                tol=tol,
                min_covar=min_covar,
                n_iter=n_iter,
                n_init=n_init,
                params=params,
                init_params=init_params)
    model = model.fit(X)
    return model

def train_dpgmm(X, n_components=3, covariance_type='diag', alpha=1.0,
                random_state=None, thresh=None, tol=0.001, verbose=False,
                min_covar=None, n_iter=10, params='wmc', init_params='wmc'):
    """
    This function trains a Infinite Gaussian Mixture Model for clustering
    :param X:
    :param n_components:
    :param covariance_type:
    :param alpha:
    :param random_state:
    :param thresh:
    :param tol:
    :param verbose:
    :param min_covar:
    :param n_iter:
    :param params:
    :param init_params:
    :return: a trained DPGMM clustering model
    """
    model = DPGMM(n_components=n_components,
                  covariance_type=covariance_type,
                  alpha=alpha,
                  random_state=random_state,
                  thresh=thresh,
                  verbose=verbose,
                  min_covar=min_covar,
                  n_iter=n_iter,
                  params=params,
                  init_params=init_params)
    model = model.fit(X)
    return model


def train_kmeans(X, n_clusters=3, init='k-means++', n_init=10,
                 max_iter=300, tol=0.0001, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True, n_jobs=1):
    """
    This functions trains a simple kmeans clustering model
    :param X:
    :param n_clusters:
    :param init:
    :param n_init:
    :param max_iter:
    :param tol:
    :param precompute_distances:
    :param verbose:
    :param random_state:
    :param copy_x:
    :param n_jobs:
    :return: trained kmeans model for clustering
    """
    model = cluster.KMeans(n_clusters=n_clusters,
                           init=init,
                           n_init=init,
                           max_iter=max_iter,
                           tol=tol,
                           precompute_distances=precompute_distances,
                           verbose=verbose,
                           random_state=random_state,
                           copy_x=copy_x,
                           n_jobs=n_jobs)
    model = model.fit(X)
    return model

def train_spectral(X, n_clusters=3, eigen_solver=None, random_state=None,
                   n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10,
                   eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                   kernel_params=None):
    """
    This functions trains Spectral clustering model
    :param X:
    :param n_clusters:
    :param eigen_solver:
    :param random_state:
    :param n_init:
    :param gamma:
    :param affinity:
    :param n_neighbors:
    :param eigen_tol:
    :param assign_labels:
    :param degree:
    :param coef0:
    :param kernel_params:
    :return: a trained Spectral Model for clustering
    """
    model = cluster.SpectralClustering(n_clusters=n_clusters,
                                       eigen_solver=eigen_solver,
                                       random_state=random_state,
                                       n_init=n_init,
                                       gamma=gamma,
                                       affinity=affinity,
                                       n_neighbors=n_neighbors,
                                       eigen_tol=eigen_tol,
                                       assign_labels=assign_labels,
                                       degree=degree,
                                       coef0=coef0,
                                       kernel_params=kernel_params)
    model = model.fit(X)
    return model

def train_agglomerative(X, n_clusters=3, affinity='euclidean',
                        connectivity=None, n_components=None,
                        compute_full_tree='auto', linkage='ward'):
    """
    This function trains hierarchical/agglomerative clustering model
    :param X:
    :param n_clusters:
    :param affinity:
    :param connectivity:
    :param n_components:
    :param compute_full_tree:
    :param linkage:
    :return: a trained hierarchical model for clustering
    """
    model = cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                            affinity=affinity,
                                            connectivity=connectivity,
                                            n_components=n_components,
                                            compute_full_tree=compute_full_tree,
                                            linkage=linkage)
    model = model.fit(X)
    return model

def train_dbscan(X, eps=0.5, min_samples=5, metric='euclidean',
                 algorithm='auto', leaf_size=30, p=None, random_state=None):
    """
    This function trains a density based spatial clustering model
    :param X:
    :param eps:
    :param min_samples:
    :param metric:
    :param algorithm:
    :param leaf_size:
    :param p:
    :param random_state:
    :return: a train DBSCAN model for clustering
    """
    model = cluster.DBSCAN(eps=eps,
                           min_samples=min_samples,
                           metric=metric,
                           algorithm=algorithm,
                           leaf_size=leaf_size,
                           p=p,
                           random_state=random_state)
    model = model.fit(X)
    return model

