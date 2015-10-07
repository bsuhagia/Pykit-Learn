# Author: Sean Dai
import cPickle
import numpy as np
import os

from nose.tools import assert_true
from pk.utils.loading import load_arff


__DIR_NAME = os.path.abspath(os.path.dirname(__file__)) + '/'

def test_load_arff():
    X, y = load_arff(__DIR_NAME + "ratings_best.arff")
    X2, y2 = cPickle.load(open(__DIR_NAME + 'correct_array.pkl','rb'))
    assert_true((X == X2).all())
    assert_true((y == y2).all())

