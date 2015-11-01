__author__ = 'Bhavesh'

from sklearn import cluster
from sklearn.mixture import GMM, DPGMM

def train_gmm(X, n_components=3, covariance_type='diag', random_state=None,
              thresh=None, tol=0.001, min_covar=0.001, n_iter=100, n_init=1,
              params='wmc', init_params='wmc'):
    """
    This function trains a gaussian mixture model for clustering
    :param X:
    :param n_components:
    :param covariance_type:
    :param random_state:
    :param thresh:
    :param tol:
    :param min_covar:
    :param n_iter:
    :param n_init:
    :param params:
    :param init_params:
    :return: a trained GMM clustering model
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

