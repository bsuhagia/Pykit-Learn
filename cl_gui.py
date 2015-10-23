# Author: Sean Dai

import shutil
import sys
import os
import cPickle
import numpy as np
import traceback

from argparse import ArgumentParser
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from pk.utils.loading import *
from pk.utils.preprocess_utils import *
import pandas.tools.plotting as pp
from pandas.tools.plotting  import radviz
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import andrews_curves
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
            return load_csv(filename, vectorize_data=True)
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

def get_pickled_dataset():
    """
    Returns X, y, and data_frame pickled files.
    """
    X = cPickle.load(open('_temp/load_X.pkl', 'r'))
    y = cPickle.load(open('_temp/load_y.pkl', 'r'))
    data_frame = cPickle.load(open('_temp/df.pkl', 'r'))
    return X, y, data_frame

def visualize_dataset(command='', plot_all=False):
    if Status.DATASET_LOADED:
        X, y, data_frame = get_pickled_dataset()
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
    elif command == 'run':
        dispatch_run(args)
    elif command == 'help':
        print help_page()
    elif command == 'quit':
        quit_gui()
    else:
        raise InvalidCommandException("{} is not a recognized command.".format(command))

def dispatch_run(args):
    parser = ArgumentParser()
    parser.add_argument('-A', dest='A', help='Select the ML algorithm to run.')
    parser.add_argument('-test_ratio', type=float, dest='test_ratio', help="Split data into training and test sets.")
    parser.add_argument('-cv', dest='cv', type=int, help='Run with cross-validation.')
    p_args = parser.parse_args(args)

    if p_args.A:
        # Run a decision tree algorithm on data
        if p_args.A.strip() == 'dt':
            print "Running decision tree algorithm on dataset..."
            X, y, _ = get_pickled_dataset()
            X_train, y_train = X, y
            X_test, y_test = X, y

            if p_args.test_ratio:
                X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                                                                    X, y, test_size=p_args.test_ratio, random_state=0)
            # Train the Decision Tree classifier
            clf = train_decision_tree(X_train, y_train)
            get_train_accuracy(clf, X_train, y_train)

            # Output metrics from train-test split
            if X_test is not None and y_test is not None:
                get_test_accuracy(clf, X_test, y_test)
            # Cross-validation score
            if p_args.cv:
                print ""
                print "Cross Validation Scores:"
                print ""
                get_cv_accuracy(clf, X_train, y_train, cv=p_args.cv)

            cm = get_confusion_matrix(clf, X_test, y_test)
            plot_confusion_matrix(cm, y=np.unique(y))

def get_train_accuracy(clf, X, y):
    print 'Train accuracy is ', clf.score(X,y)*100, ' %'
    return clf.score(X,y)

def get_test_accuracy(clf, X, y):
    print 'Test accuracy is ', clf.score(X, y)*100, ' %'
    return clf.score(X, y)

def get_cv_accuracy(clf, X, y, cv=10):
    scores = cross_validation.cross_val_score(clf, X, y, cv=cv)
    print 'Scores are : ', scores
    avg = scores.mean()
    print 'Average accuracy is : ', avg, ' (+/- ' , scores.std()*2, ')'
    return scores, avg

def benchmark(training_func):
    pass

def setup():
    if not os.path.exists("_temp/"):
        os.mkdir("_temp/")

def quit_gui():
    shutil.rmtree("_temp")
    sys.exit(1)

def train_decision_tree(X, y, criterion='gini',splitter='best', max_depth=None,
                        min_samples_split=2, min_samples_leaf=1,
                        max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None):
    """
    Builds a decision tree model

    Returns:
     clf: Fitted Decision tree classifier object
    """
    clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 max_features=max_features, random_state=random_state, max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(X, y)
    return clf

def get_confusion_matrix(clf, X, true_y):
    predicted_y = clf.predict(X)
    matrix = confusion_matrix(true_y, predicted_y)
    print 'Confusion Matrix is: \n', matrix
    return matrix

def plot_confusion_matrix(cm, y, title='Confusion matrix', cmap = plt.cm.Blues, continous_class = False):
    if continous_class == True:
        return None
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(y)))
    plt.xticks(tick_marks, np.unique(y), rotation=45)
    plt.yticks(tick_marks, np.unique(y))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

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
    run -A [alg]            Runs the ML alg on the dataset.
                            Options for [alg]:
                                dt (Decision Tree)
                                Eg. "run -A dt -test_ratio .3 -cv 5"
        -test_ratio [0-1]   User can specify the test-train ratio.
        -cv [int]           Enables k-fold cross validation.
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