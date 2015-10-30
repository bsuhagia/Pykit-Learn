# Author: Sean Dai
import logging
import os

from numpy.testing import assert_array_equal
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from pk.utils.preprocess_utils import *
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from pk.utils.loading import load_csv

__DIR_NAME = os.path.abspath(os.path.dirname(__file__)) + '/'

def test_standardize():
    digits = load_digits()
    X = digits.data
    assert_true((standardize(X) == StandardScaler().fit_transform(X)).all())

def test_normalize_data():
    boston = load_boston()
    X = boston.data
    assert_true((normalize_data(X) == Normalizer().fit_transform(X)).all())

def test_remove_incomplete_examples():
    X, y, _ = load_csv(__DIR_NAME + 'blank.csv')
    assert len(X) == len(y)
    X, y = remove_incomplete_examples(X, y, '?')

    exp_X = np.array([[1,2,3,4],
                     [2,3,4,5],
                     [2,3,4,5],
                     [1,2,3,4]])
    exp_X = exp_X.astype('str')
    exp_y = np.array(['good', 'good', 'good', 'good'])
    assert_array_equal(X, exp_X)
    assert_array_equal(y, exp_y)
