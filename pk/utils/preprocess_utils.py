import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

def standardize(X, axis=0, with_mean=True, with_std=True, copy=True):
    """
    Standardize a dataset along any axis.

    Args:
        X: numpy feature array of size (n_examples, n_features)
        axis: axis to compute mean and stds along.
        with_mean: if True, center data before scaling
        with_std: if True, scale to unit variance
        copy: if False, do inplace normalization and avoid copying array

    Returns:
        changed_X: mean-shifted X with unit variance
    """
    return scale(X, axis, with_mean, with_std, copy)

def normalize_data(X, norm='l2', axis=1, copy=True):
    """
    Scale input vectors to unit norm.

    Args:
        X: numpy feature array with shape (n_samples, n_features)
        norm: the norm to use
        axis: axis along which to normalize
        copy: if False, do inplace row normalization
    Returns:
        A normalized numpy array.
    """
    return normalize(X, norm, axis, copy)

def binarize():
    pass
def remove_incomplete_examples(missing_char="?"):
    """
    Removes examples with missing/incomplete features.

    Args:
        missing_char: Placeholder for intended value
    Returns:
        Numpy feature array with bad examples removed
    """
