# Author: Sean Dai

import shutil
import sys
import os
import cPickle
import numpy as np
import traceback

from optparse import OptionParser
from collections import Counter
from pk.utils.loading import *
from pk.utils.preprocess_utils import *
import pandas.tools.plotting as pp
from pandas.tools.plotting  import radviz
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import andrews_curves
import matplotlib.pyplot as plt
import seaborn as sns

class Status(object):
    DATASET_LOADED = False
    FILENAME = None
    EXTENSION = None

class InvalidCommandException(Exception):
    def __init__(self, message, errors=None):
        super(InvalidCommandException, self).__init__(message)
        self.errors = errors

def _load_file(filename):
        extension = filename[filename.rfind('.'):]
        if (extension == '.csv'):
            return load_csv(filename)
        elif (extension == '.arff'):
            return load_arff(filename)
        elif (extension == '.xls' or extension == '.xlsx'):
            return load_excel(filename)
        else:
            raise IOError('{} is not a valid filename!'.format(filename))

def load_file(filename):
    X, y, data_frame = _load_file(filename)
    pickle_files(*[(X, 'load_X.pkl'), (y, 'load_y.pkl'), (data_frame, 'df.pkl')])
    Status.DATASET_LOADED = True
    Status.FILENAME = filename
    Status.EXTENSION = filename[filename.rfind('.')]

    print 'Feature Array:\n %s' % X
    print 'Target classifications:\n %s' % y

def pickle_files(*args):
    """
    Saves a list of files to _temp directory

    Input: List of tuples in form (obj, filename_to_save)
    """
    for obj, filename in args:
        with open("_temp/" + filename, 'wb') as f:
            cPickle.dump(obj, f)

def visualize_dataset(command='', plot_all=False):
    if Status.DATASET_LOADED:
        X = cPickle.load(open('_temp/load_X.pkl', 'r'))
        y = cPickle.load(open('_temp/load_y.pkl', 'r'))
        data_frame = cPickle.load(open('_temp/df.pkl', 'r'))
        class_name = data_frame.dtypes.index[-1]

        if command == 'class_frequency' or plot_all:
            plot_class_frequency_bar(y)
        if command == 'feature_matrix':
            plot_feature_matrix(data_frame)
        if command == 'radial' or plot_all:
            plot_radial(data_frame, class_name)
        if command == 'andrews' or plot_all:
            plot_andrews(data_frame, class_name)

    else:
        raise Exception("Can't visualize an unloaded dataset!")

def plot_class_frequency_bar(target, bar_width=.35):
    # Get the frequency of each class label
    classes = np.unique(target)
    target_counts = Counter(target)

    # Plot the bar chart of class frequencies
    fig, ax = plt.subplots()
    ind = np.arange(len(classes))
    ax.set_xticks(ind)
    rects = ax.bar(ind, target_counts.values(), width=bar_width, align='center')
    ax.set_title(Status.FILENAME)
    ax.set_ylabel('Frequency')

    ax.set_xticklabels(target_counts.keys())
    fig.show()

def plot_feature_matrix(data_frame):
    # Plot the matrix of feature-feature pairs
    g = sns.PairGrid(data_frame)
    g.map(plt.scatter)
    plt.show(block=False)

def plot_radial(data_frame, class_name):
    fig = plt.figure()
    radviz(data_frame, class_name)
    plt.show(block=False)

def plot_andrews(data_frame, class_name):
    fig = plt.figure()
    andrews_curves(data_frame, class_name)
    plt.show(block=False)

def plot_scatter_matrix(data_frame):
    scatter_matrix(data_frame, alpha=0.2, figsize=(10,10), diagonal='kde')
    plt.show(block=False)

def process(line):
    tokens = tuple(line.split(' '))
    command, args = tokens[0], tokens[1:]
    if command == 'load':
        load_file(*args)
    elif command == 'plot_frequency':
        visualize_dataset('class_frequency')
    elif command == 'plot_matrix':
        visualize_dataset('feature_matrix')
    elif command == 'plot_radial':
        visualize_dataset('radial')
    elif command == 'plot_andrews':
        visualize_dataset('andrews')
    elif command == 'plot_scatter_matrix':
        visualize_dataset('scatter_matrix')
    elif command == 'visualize':
        visualize_dataset(plot_all=True)
    elif command == 'help':
        print help_page()
    elif command == 'quit':
        quit_gui()
    else:
        raise InvalidCommandException("{} is not a recognized command.".format(command))

def setup():
    if not os.path.exists("_temp/"):
        os.mkdir("_temp/")

def quit_gui():
    shutil.rmtree("_temp")
    sys.exit(1)

def help_page():
    output_page = """
Pykit-Learn Command Line GUI
--------------------------------
Commands:
    The following commands are available:

    load [file]             Loads the dataset at the path specified by [file].
                            No quotes "" to file!
    plot_andrews            Plots an Andrews curve of the dataset.

    plot_frequency          View the frequency of each class label.
    plot_matrix             Generate a matrix plot of feature-feature
                            relationships.
    plot_radial             Plot a radial chart of the dataset.
    visualize               Plots all possible visualizations for input data.
    help                    Provides a help screen of available commands.
    quit                    Quits the command line GUI.
    """
    return output_page

def main():
    """
    To run, type "python cl_gui.py".
    """
    print "Welcome to the command-line version of Pykit-Learn!"
    print "Type 'help' for a list of available commands"
    setup()

    while True:
        try:
            input_line = raw_input(">> ")
            process(input_line.strip())
        except IOError as ioe:
            print ioe.message
        except InvalidCommandException as inv:
            print inv.message
        except Exception as e:
            traceback.print_exc()
        except KeyboardInterrupt:
            quit_gui()

if __name__ == "__main__":
    main()