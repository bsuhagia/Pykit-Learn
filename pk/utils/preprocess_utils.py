import numpy as np
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder

from pk.utils.loading import is_number

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

def encode_labels(X):
    """
    Converts categorical feature columns to a numerical values 0 - num_features.

    Arguments:
        X: feature array

    Returns:
        Feature array with categorical columns replaced with numbers.
    """
    # Gets feature labels and stores them in a dict.
    feature_dict = { i:X[:, i, np.newaxis] for i in xrange(len(X[0])) }
    for i in feature_dict:
        if not is_number(X[0, i]):
            feature_dict[i] = LabelEncoder().fit_transform(feature_dict[i])

    return np.array(feature_dict.values()).T

def remove_incomplete_examples(X, y, missing_char="?"):
    """
    Removes examples with missing/incomplete features.

    Args:
        missing_char: Placeholder for intended value
    Returns:
        Numpy feature array X with bad examples removed
        target classes with bad examples removed
    """
    row_ind, _ = np.where(X == missing_char)
    row_ind = np.unique(row_ind)
    valid_rows = np.delete(np.arange(len(X)), row_ind)
    return X[valid_rows, :], y[valid_rows]