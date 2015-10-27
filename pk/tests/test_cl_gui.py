"""This module tests the command line GUI.
    Author: Sean Dai
"""
import cl_gui
from nose.tools import nottest
from nose.tools import assert_raises
from nose.tools import assert_true
import os

def setup():
    os.chdir(os.path.abspath(os.path.join(__file__, '../../..')))
    cl_gui.setup()

def td():
    with assert_raises(SystemExit):
        cl_gui.quit_gui()

@nottest
def get_test_accuracy():
    pass

def test_visualize_iris():
    setup()
    cl_gui.process('load pk/tests/iris.csv')
    cl_gui.process('visualize --suppress')
    temp_files = os.listdir('_temp/')
    assert_true('plot_andrews.png' in temp_files)
    assert_true('plot_frequency.png' in temp_files)
    assert_true('plot_radial.png' in temp_files)
    td()

def test_preprocess_flow():
    setup()
    cl_gui.process('load pk/tests/iris2.csv')
    cl_gui.process('preprocess -std -norm')
    cl_gui.process('plot_radial --suppress')
    temp_files = os.listdir('_temp/')
    assert_true('plot_radial.png' in temp_files)
    td()

def test_run_decision_tree():
    setup()
    cl_gui.process('load pk/tests/iris2.csv')
    cl_gui.process('run -A dt -test_ratio .5 -cv 15')
    td()