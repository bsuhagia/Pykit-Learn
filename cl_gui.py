""" This module contains an executable command-line version of the Pykit-Learn
GUI.
    Author: Sean Dai
"""

import cPickle
import logging
import os
import shutil
import sys
import traceback
from argparse import ArgumentParser
from collections import Counter

from pandas.tools.plotting import radviz
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import andrews_curves
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from pk.utils.loading import *
from pk.utils.preprocess_utils import *


class Status(object):
    DATASET_LOADED = False
    FILENAME = ''
    EXTENSION = None
    USER_QUIT = 'user_quit'


class InvalidCommandException(Exception):
    def __init__(self, message, errors=None):
        super(InvalidCommandException, self).__init__(message)
        self.errors = errors


def _load_file(filename):
    extension = filename[filename.rfind('.'):]
    if extension == '.csv':
        return load_csv(filename, vectorize_data=True)
    elif extension == '.arff':
        return load_arff(filename)
    elif extension == '.xls' or extension == '.xlsx':
        return load_excel(filename)
    else:
        raise IOError('{} is not a valid filename!'.format(filename))


def load_file(filename):
    """
    Function to load a dataset file.
    """
    X, y, data_frame = _load_file(filename)
    pickle_files([(X, 'load_X.pkl'), (y, 'load_y.pkl'), (data_frame, 'df.pkl')])

    # Update appropriate status flags.
    Status.DATASET_LOADED = True
    Status.FILENAME = filename
    Status.EXTENSION = filename[filename.rfind('.')]

    print 'Feature Array:\n %s' % X
    print 'Target classifications:\n %s' % y


def load_random():
    """
    Generates a random dataset with 100 samples, 2 features, and 3 classes.
    """
    X, y, df = generate_random_points()
    pickle_files([(X, 'load_X.pkl'), (y, 'load_y.pkl'), (df, 'df.pkl')])

    # Update appropriate status flags.
    Status.DATASET_LOADED = True

    print 'Feature Array:\n %s' % X
    print 'Target classifications:\n %s' % y


def pickle_files(files_to_save):
    """
    Saves a list of files to _temp directory

    Input: List of tuples in form (obj, filename_to_save)
    """
    for obj, filename in files_to_save:
        with open("_temp/" + filename, 'wb') as f:
            cPickle.dump(obj, f)


def get_pickled_dataset():
    """
    Returns X, y, and data_frame pickled files.
    """
    f1 = open('_temp/load_X.pkl', 'r')
    f2 = open('_temp/load_y.pkl', 'r')
    f3 = open('_temp/df.pkl', 'r')

    X = cPickle.load(f1)
    y = cPickle.load(f2)
    data_frame = cPickle.load(f3)

    f1.close()
    f2.close()
    f3.close()
    return X, y, data_frame


def update_feature_array(changed_X):
    with open('_temp/load_X.pkl', 'wb') as f:
        cPickle.dump(changed_X, f)
    with open('_temp/df.pkl', 'wb') as f:
        cPickle.dump(pd.DataFrame(changed_X), f)


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
        raise InvalidCommandException("Can't visualize an unloaded dataset!")


def plot_class_frequency_bar(target, bar_width=.35):
    # Get the frequency of each class label
    classes = np.unique(target)
    target_counts = Counter(target)

    # Plot the bar chart of class frequencies
    fig, ax = plt.subplots()
    ind = np.arange(len(classes))
    ax.set_xticks(ind)
    ax.bar(ind, target_counts.values(), width=bar_width, align='center')
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
    plt.figure()
    radviz(data_frame, class_name)
    plt.show(block=False)


def plot_andrews(data_frame, class_name):
    plt.figure()
    andrews_curves(data_frame, class_name)
    plt.show(block=False)


def plot_scatter_matrix(data_frame):
    scatter_matrix(data_frame, alpha=0.2, figsize=(10, 10), diagonal='kde')
    plt.show(block=False)


def dispatch_preprocess(args):
    if not Status.DATASET_LOADED:
        raise InvalidCommandException("Can't preprocess an unloaded dataset!")

    parser = ArgumentParser()
    parser.add_argument('-std', dest='std', action='store_true',
                        help='Standardize the feature array.')
    parser.add_argument('-norm', dest='norm', action='store_true',
                        help="Normalize the values of each feature.")
    p_args = parser.parse_args(args)
    print p_args

    if p_args.std:
        print "Standardizing feature array..."
        X, y, _ = get_pickled_dataset()
        new_X = standardize(X)
        print new_X
        update_feature_array(new_X)
    if p_args.norm:
        print "Normalizing feature array..."
        X, y, _ = get_pickled_dataset()
        new_X = normalize_data(X)
        print new_X
        update_feature_array(new_X)


def dispatch_run(args):
    # Build parser for "run" flags
    parser = ArgumentParser()
    parser.add_argument('-A', dest='A', help='Select the ML algorithm to run.')
    parser.add_argument('-test_ratio', type=float, dest='test_ratio',
                        help="Split data into training and test sets.")
    parser.add_argument('-cv', dest='cv', type=int,
                        help='Run with cross-validation.')
    p_args = parser.parse_args(args)

    # Process the passed in arguments
    if p_args.A:
        # Run a decision tree algorithm on data
        if p_args.A.strip() == 'dt':
            print "Running decision tree algorithm on dataset..."
            X, y, _ = get_pickled_dataset()
            X_train, y_train = X, y
            X_test, y_test = X, y

            # Split the original dataset to training & testing sets
            if p_args.test_ratio:
                X_train, X_test, y_train, y_test = cross_validation.train_test_split(
                    X, y, test_size=p_args.test_ratio,
                    random_state=0)
            # Train the Decision Tree classifier
            clf = train_decision_tree(X_train, y_train)
            get_train_accuracy(clf, X_train, y_train)

            # Output metrics from train-test split
            if X_test is not None and y_test is not None:
                get_test_accuracy(clf, X_test, y_test)

            # Get cross-validation score(s)
            if p_args.cv:
                print ""
                print "Cross Validation Scores:"
                get_cv_accuracy(clf, X_train, y_train, cv=p_args.cv)

            # Plot the confusion matrix
            cm = get_confusion_matrix(clf, X_test, y_test)
            plot_confusion_matrix(cm, y=np.unique(y))


def train_decision_tree(X, y, criterion='gini', splitter='best', max_depth=None,
                        min_samples_split=2, min_samples_leaf=1,
                        max_features=None, random_state=None,
                        max_leaf_nodes=None):
    """
    Builds a decision tree model.

    Returns:
     clf: Fitted Decision tree classifier object
    """
    clf = DecisionTreeClassifier(criterion=criterion,
                                 splitter=splitter,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 max_features=max_features,
                                 random_state=random_state,
                                 max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(X, y)
    return clf


def get_confusion_matrix(clf, X, true_y):
    predicted_y = clf.predict(X)
    matrix = confusion_matrix(true_y, predicted_y)
    print 'Confusion Matrix is: \n%s' % matrix
    return matrix


def plot_confusion_matrix(cm, y, title='Confusion matrix', cmap=plt.cm.Blues,
                          continuous_class=False):
    if continuous_class:
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
    plt.show(block=False)


def get_train_accuracy(clf, X, y):
    print 'Train accuracy is %f%%' % (clf.score(X, y) * 100)
    return clf.score(X, y)


def get_test_accuracy(clf, X, y):
    print 'Test accuracy is %f%%' % (clf.score(X, y) * 100)
    return clf.score(X, y)


def get_cv_accuracy(clf, X, y, cv=10):
    scores = cross_validation.cross_val_score(clf, X, y, cv=cv)
    print 'Scores: ' + ', '.join(map(str, scores))
    avg = scores.mean()
    print 'Average accuracy: %f (+/- %f)' % (avg, scores.std() * 2)
    return scores, avg


def benchmark(X, y, training_func, *args, **kwargs):
    clf = training_func(X, y, *args, **kwargs)
    get_train_accuracy(clf, X, y)
    get_test_accuracy(clf, X, y)


def setup():
    # Create temporary directory for storing serialized objects.
    if not os.path.exists("_temp/"):
        os.mkdir("_temp/")

    # Configure log file for the application.
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        filename='cl_gui.log')
    logging.info("Starting application...")

    # Ignore any warnings issued by third-party modules
    import warnings

    warnings.filterwarnings("ignore")

    # Code snippet for recalling previous commands with the
    # 'up' and 'down' arrow keys.
    import rlcompleter
    import atexit
    import readline

    histfile = os.path.join(os.environ['HOME'], '.pythonhistory')
    try:
        readline.read_history_file(histfile)
    except IOError:
        pass
    atexit.register(readline.write_history_file, histfile)

    # Tab completion for GUI commands
    def completer(text, state):
        commands = ['load', 'load_random', 'plot_andrews', 'plot_frequency',
                    'plot_matrix', 'plot_radial', 'preprocess', 'run',
                    'visualize', 'help', 'quit']
        for dirname, dirnames, filenames in os.walk('.'):
            if '.git' in dirnames:
                # don't go into any .git directories.
                dirnames.remove('.git')
            # Add path to subdirectories
            commands.extend([os.path.join(dirname, sub_dir) for sub_dir in dirnames])
            # Add path to all filenames in subdirectories.
            commands.extend([os.path.join(dirname, filename) for filename in filenames])
            # Remove './' header in file strings.
            commands = [cmd.strip('./') for cmd in commands]

        options = [i for i in commands if i.startswith(text)]
        try:
            return options[state]
        except IndexError:
            return None

    readline.set_completer(completer)

    # Bind tab completer to specific platforms
    if 'libedit' in readline.__doc__:
        readline.parse_and_bind("bind -e")
        readline.parse_and_bind("bind '\t' rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    del histfile, readline, rlcompleter, warnings


def quit_gui():
    shutil.rmtree("_temp")
    logging.info("Quitting application...")
    sys.exit(Status.USER_QUIT)


def help_page():
    output_page = """
Pykit-Learn Command Line GUI
--------------------------------
Commands:
    The following commands are available:

    load [file]             Loads the dataset at the path specified by [file].
                            No quotes "" around filename!
    load_random             Load a randomly generated dataset with 3 classes.
    plot_andrews            Plots an Andrews curve of the dataset.

    plot_frequency          View the frequency of each class label.
    plot_matrix             Generate a matrix plot of feature-feature
                            relationships.
    plot_radial             Plot a radial chart of the dataset.
    preprocess [flags]      Preprocesses a dataset. Flags are
                                -std Standardize to mean 0 and variance 1
                                -norm Normalize each feature to range [0,1]
                                Eg. "preprocess -std"
    run                     Runs the ML alg on the loaded dataset.
        -A [alg]            REQUIRED flag! Options for [alg]:
                                dt = (Decision Tree)
        -test_ratio [0-1]   User can specify the test-train ratio.
        -cv [int]           Enables k-fold cross validation.
                            Example: "run -A dt -test_ratio .3 -cv 5"
    visualize               Plots all possible visualizations for input data.
    help                    Provides a help screen of available commands.
    quit                    Quits the command line GUI.
    """
    return output_page


def process(line):
    tokens = tuple(line.split(' '))
    command, args = tokens[0], tokens[1:]

    # Select the appropriate function to call
    if command == 'load':
        load_file(*args)
    elif command == 'load_random':
        load_random()
    elif command == 'preprocess':
        dispatch_preprocess(args)
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
    elif command == '':
        return
    else:
        raise InvalidCommandException(
            "{} is not a recognized command.".format(command))


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
        except Exception:
            traceback.print_exc()
        except SystemExit as se:
            if str(se.message) == Status.USER_QUIT:
                return
            else:
                print se.message
        except KeyboardInterrupt:
            quit_gui()


if __name__ == "__main__":
    main()
