# Author: Sean Dai
import os

from numpy.testing import assert_array_equal
from nose.tools import assert_true
from numpy.testing import assert_array_almost_equal

from pk.utils.preprocess_utils import PreprocessingEngine
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

from pk.utils.loading import load_csv
import numpy as np

__DIR_NAME = os.path.abspath(os.path.dirname(__file__)) + '/'
pe = PreprocessingEngine()

def test_standardize():
    digits = load_digits()
    X = digits.data
    assert_true((pe.standardize(X) == StandardScaler().fit_transform(X)).all())

def test_normalize_data():
    boston = load_boston()
    X = boston.data
    assert_true((pe.normalize_data(X) == Normalizer().fit_transform(X)).all())

def test_remove_incomplete_examples():
    X, y, _ = load_csv(__DIR_NAME + 'blank.csv')
    assert len(X) == len(y)
    X, y = pe.remove_incomplete_examples(X, y, '?')

    exp_X = np.array([[1,2,3,4],
                     [2,3,4,5],
                     [2,3,4,5],
                     [1,2,3,4]])
    exp_X = exp_X.astype('str')
    exp_y = np.array(['good', 'good', 'good', 'good'])
    assert_array_equal(X, exp_X)
    assert_array_equal(y, exp_y)

def test_label_encoder():
    X = np.array([['a','b',1], ['a','a',11], ['b','b',13], ['c', 'c', 100]])
    expX = np.array([[0, 1, 1],
                     [0, 0, 11],
                     [1, 1, 13],
                     [2, 2, 100]])
    assert_array_almost_equal(pe.convert_to_float_array(pe.encode_labels(X)),
                              expX)

def test_binarize():
    y = ['a', 'b', 'c', 'a']
    exp_y = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],
             [1, 0, 0]]
    assert_array_almost_equal(pe.binarize(y), exp_y)

def test_inpute_missing_values():
    X = np.array([[1,2,'NaN'], [3,'NaN',5], [1,2,3]])
    X = pe.encode_labels(X)
    X = pe.impute_missing_values(X, missing_values='NaN')
    exp_X = np.array([[ 1.,  2.,  4.],
                      [ 3.,  2.,  5.],
                      [ 1.,  2.,  3.]])
    assert_array_almost_equal(X, exp_X)