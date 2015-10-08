# Author: Sean Dai
import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from scipy.io.arff import loadarff

def _load_arff(filename):
    dataset = loadarff(open(filename,'r'))
    features = dataset[1].names()
    class_attr = features[-1]
    y = np.array(dataset[0][class_attr])
    X = np.array(dataset[0][features[:-1]])
    X = np.array(map(list, X))
    return X, y, features

def is_numeric_type(array):
    """
    Checks if the array's datatype is a number data type.

    Args:
        array: numpy array

    Returns:
        True if array.dtype is type float, int, uint, complex, or bool
        Otherwise, we say it's a string.
    """
    numeric_dtypes = [np.bool_]
    numeric_strings = set(['uint', 'complex', 'float', 'int'])
    for dtype, entries in np.sctypes.items():
        if dtype in numeric_strings:
            numeric_dtypes.extend(entries)
    return array.dtype.type in numeric_dtypes

def vectorize_categorical_data(X, y, features):
    """
    One-hot encoding for categorical attributes in the feature array.

    Args:
        X: (num_examples * num_features) numpy array of all the examples
        y: the class labels of size (1 * num_examples)
        features: list of feature names

    Returns:
        X: new numpy array with all categorical labels becoming 1-hot encoded
        y: class labels, changed to 1-hot if labels were categorical
    """
    vec = DictVectorizer()
    assert (len(features) - 1) == len(X[0])

    """
    Create a dictionary for each example with the feature name as the key.
    DictVectorizer requires feature arrays to be represented as a list
    of dict objects. Each element of the list is 1 feature vector example from
    the dataset.
    """
    measurements = []
    for ex in X:
        ex_dict = dict(zip(features, ex.tolist()))
        measurements.append(ex_dict)
    measurements = _convert_dict_values_to_num(measurements)

    if not is_numeric_type(y):
        y = _convert_target_to_num(y)

    X = vec.fit_transform(measurements, y).toarray()
    return X, y


def _convert_dict_values_to_num(examples):
    """
    Convert only the numeric values formatted as strings to actual
    numeric datatypes in the feature array of dicts.

    examples - list<dict>
    """
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError as e:
            return False

    new_examples = examples[:]
    for dt in new_examples:
        for key in dt:
            value = dt[key]
            if is_number(value):
                dt[key] = float(value)
    return new_examples

def _convert_target_to_num(target):
    """
    Convert only the numeric values formatted as strings to actual
    numeric datatypes in the feature array of dicts.

    target - nd.array of class values

    Returns:
        converted target array to float dtype
    """
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError as e:
            return False

    if all(map(is_number, target)):
        return target.astype(float)
    else:
        return target

def load_arff(filename):
    """
    Loads .arff dataset files.

    Args:
        filename: str

    Returns:
        X : a (num_examples * num_features) numpy array of examples X
        y : the class labels y of size (1, num_examples)
        features : name of each feature (list<str>)
    """
    X, y, features = _load_arff(filename)

    # For categorical data, we want the feature label names
    # in order to create a 1-hot encoding of the categorical
    # values in our feature array of examples.
    if not is_numeric_type(X):
        return vectorize_categorical_data(X, y, features)
    else:
        return X, y