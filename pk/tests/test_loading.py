# Author: Sean Dai
import cPickle
import logging
import os

from numpy.testing import assert_array_equal
from nose.tools import assert_true
from nose.plugins.attrib import attr
from numpy.testing import assert_array_almost_equal
from pandas.util.testing import assert_frame_equal
from pk.utils.loading import *


__DIR_NAME = os.path.abspath(os.path.dirname(__file__)) + '/'


def test_load_arff():
    X, y, _ = load_arff(__DIR_NAME + "ratings_best.arff")
    X2, y2 = cPickle.load(open(__DIR_NAME + 'correct_array.pkl', 'r'))
    assert_true((X == X2).all())
    assert_true((y == y2).all())


def test_load_arff_categorical():
    X, y, _ = load_arff(__DIR_NAME + "credit-g.arff")
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
    X, y, _ = load_arff(__DIR_NAME + "credit-g.arff", vectorize_data=False)
    correct_list = ["'<0'", '6.0', "'critical/other existing credit'", 'radio/tv', '1169.0',
                    "'no known savings'", "'>=7'", '4.0', "'male single'", 'none', '4.0',
                    "'real estate'", '67.0', 'none' ,'own' ,'2.0', 'skilled', '1.0', 'yes', 'yes']
    assert_array_equal(X[0], correct_list)

def test_load_csv():
    filename = __DIR_NAME + 'iris.csv'
    X, y, _ = load_csv(filename)
    expX = [[5.8,4,1.2,0.2],
            [5.9,3,4.2,1.5],
            [6.5,3.2,5.1,2]]
    expY = ['setosa', 'versicolor', 'virginica']
    assert_array_equal(X, expX)
    assert_array_equal(y, expY)

def test_load_excel():
    filename = __DIR_NAME + 'Wine.xls'
    X, y, _ = load_excel(filename)
    expX = np.array([[ 1.42300000e+01,   1.71000000e+00,   2.43000000e+00,
                       1.56000000e+01,   1.27000000e+02,   2.80000000e+00,
                       3.06000000e+00,   2.80000000e-01,   2.29000000e+00,
                       5.64000000e+00,   1.04000000e+00,   3.92000000e+00,
                       1.06500000e+03],
                    [  1.23700000e+01,   9.40000000e-01,   1.36000000e+00,
                       1.06000000e+01,   8.80000000e+01,   1.98000000e+00,
                       5.70000000e-01,   2.80000000e-01,   4.20000000e-01,
                       1.95000000e+00,   1.05000000e+00,   1.82000000e+00,
                       5.20000000e+02],
                    [  1.28600000e+01,   1.35000000e+00,   2.32000000e+00,
                       1.80000000e+01,   1.22000000e+02,   1.51000000e+00,
                       1.25000000e+00,   2.10000000e-01,   9.40000000e-01,
                       4.10000000e+00,   7.60000000e-01,   1.29000000e+00,
                       6.30000000e+02]])
    expY = ['A', 'B', 'C']
    assert_array_almost_equal(X, expX)
    assert_array_equal(y, expY)

def test_generate_random():
    np.random.seed(42)
    X, y, df = generate_random_points(5)
    expX = np.array([[-0.92998481,  9.78172086],
                       [ 4.88184111,  0.05988944],
                       [-2.97867201,  9.55684617],
                       [-8.60454502, -7.44239712],
                       [ 4.17646114,  1.50743993]])
    expY = np.array([0, 1, 0, 2, 1])
    exp_df = pd.DataFrame(np.hstack((expX,expY[:, np.newaxis])))

    assert_array_almost_equal(X, expX)
    assert_array_equal(y, expY)
    assert_frame_equal(df, exp_df)

@attr('slow')
def test_mldata():
    dl = DatasetIO()
    X, y, df = dl.load_from_mldata('iris')

# test_load_arff()
# test_load_arff_categorical()
# test_vectorize()
# test_vectorize_numeric()
# test_load_categorical_no_vectorize()
# test_load_excel()
# test_generate_random()