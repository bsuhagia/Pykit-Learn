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

def test_standardize():
    digits = load_digits()
    X = digits.data
    assert_true((standardize(X) == StandardScaler().fit_transform(X)).all())

def test_normalize_data():
    boston = load_boston()
    X = boston.data
    assert_true((normalize_data(X) == Normalizer().fit_transform(X)).all())