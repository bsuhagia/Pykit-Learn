# Author: Sean Dai
import cPickle
import logging
import os

from numpy.testing import assert_array_equal
from nose.tools import assert_true

from pk.utils.loading import *


__DIR_NAME = os.path.abspath(os.path.dirname(__file__)) + '/'


def test_load_arff():
    X, y = load_arff(__DIR_NAME + "ratings_best.arff")
    X2, y2 = cPickle.load(open(__DIR_NAME + 'correct_array.pkl', 'r'))
    assert_true((X == X2).all())
    assert_true((y == y2).all())


def test_load_arff_categorical():
    X, y = load_arff(__DIR_NAME + "credit-g.arff")
    print X, y
    logging.info((X, y))


def test_vectorize():
    X = np.array([['a', 1], ['b', 2], ['a', 1]])
    y = np.array(['0', '1', '0'])
    features = ['f1', 'f2', 'class']
    X2, y2 = vectorize_categorical_data(X, y, features)
    exp_vec_X = np.array([[1., 0., 1.],
                          [0., 1., 2.],
                          [1., 0., 1.]])
    assert_array_equal(exp_vec_X, X2)


def test_vectorize_numeric():
    X = np.array([[0, 1, 3, 4], [2, 1, 1, 1], [4, 55, 2, 1]])
    y = np.array([0, 1, 0, 1])
    features = ['num1', 'num2', 'num3', 'num4', 'class']
    X2, y2 = vectorize_categorical_data(X, y, features)
    exp_vec_X = np.array([[0., 1., 3., 4.],
                          [2., 1., 1., 1.],
                          [4., 55., 2., 1.]])
    exp_vec_y = np.array([0, 1, 0, 1])
    assert_array_equal(X2, exp_vec_X)
    assert_array_equal(y2, exp_vec_y)

def test_vectorize_bool_numeric():
    X = np.array([[0, 1, 3, True], [2, 1, 1, False], [4, 55, 2, True]])
    y = np.array([0, 1, 0, 1])
    features = ['num1', 'num2', 'num3', 'num4', 'class']
    X2, y2 = vectorize_categorical_data(X, y, features)
    exp_vec_X = np.array([[0., 1., 3., 1.],
                          [2., 1., 1., 0.],
                          [4., 55., 2., 1.]])
    exp_vec_y = np.array([0, 1, 0, 1])
    assert_array_equal(X2, exp_vec_X)
    assert_array_equal(y2, exp_vec_y)

def test_vectorize_bool_only():
    X = np.array([[False, True], [False, False], [True, True]])
    y = np.array([0, 1, 0])
    features = ['bool1', 'bool2', 'class']
    X2, y2 = vectorize_categorical_data(X, y, features)
    exp_vec_X = np.array([[0, 1],
                          [0, 0],
                          [1, 1]])
    exp_vec_y = np.array([0, 1, 0])
    assert_array_equal(X2, exp_vec_X)
    assert_array_equal(y2, exp_vec_y)

def test_load_categorical_no_vectorize():
    X, y = load_arff(__DIR_NAME + "credit-g.arff", vectorizeData=False)
    correct_list = ["'<0'", '6.0', "'critical/other existing credit'", 'radio/tv', '1169.0',
                    "'no known savings'", "'>=7'", '4.0', "'male single'", 'none', '4.0',
                    "'real estate'", '67.0', 'none' ,'own' ,'2.0', 'skilled', '1.0', 'yes', 'yes']
    assert_array_equal(X[0], correct_list)

# test_load_arff()
# test_load_arff_categorical()
# test_vectorize()
# test_vectorize_numeric()
# test_load_categorical_no_vectorize()