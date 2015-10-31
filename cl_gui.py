""" This module contains an executable command-line version of the Pykit-Learn
GUI.
    Author: Sean Dai
"""

import cPickle
import logging
import multiprocessing
import os
import shutil
import sys
import traceback
from argparse import ArgumentParser
from collections import Counter
from glob import glob
from os.path import join


from pandas.tools.plotting import radviz
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import andrews_curves
from sklearn import cross_validation



import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from pk.utils.loading import *
from pk.utils.preprocess_utils import *
from pk.utils.prygress import progress
from pk.utils.classification_utils import *
from pk.utils.performance_utils import  *

class Status(object):
    DATASET_LOADED = False
    FILENAME = ''
    EXTENSION = None
    TEMP_DIR = '_temp/'
    USER_QUIT = 'user_quit'
    RADIAL_NAME = 'plot_radial.png'
    SCM_NAME = 'plot_scatter_matrix.png'
    FREQ_NAME = 'plot_frequency.png'
    FM_NAME = 'plot_feature_matrix.png'
    ANDREWS_NAME = 'plot_andrews.png'
    TD_NAME = 'plot_2d.png'
    FINISH_PLOTS = False
    PLOT_COMMANDS = {'plot_frequency', 'plot_feature_matrix', 'plot_radial',
                     'plot_andrews', 'plot_scatter_matrix', 'plot_2d'}
    ALL_COMMANDS = list(PLOT_COMMANDS) + ['load', 'load_random', 'preprocess',
                                          'run', 'visualize', 'help', 'quit',
                                          'see_images']


class InvalidCommandException(Exception):
    def __init__(self, message, errors=None):
        super(InvalidCommandException, self).__init__(message)
        self.errors = errors

def _load_file(filename):
    loader = DatasetIO()
    return loader.load_file(filename)

def load_file(filename):
    """
    Function to load a dataset file.
    """
    X, y, df = _load_file(filename)

    loader = DatasetIO()
    loader.pickle_files([(X, 'load_X.pkl'), (y, 'load_y.pkl'), (df, 'df.pkl')],
                        Status.TEMP_DIR)

    # Update appropriate status flags.
    Status.DATASET_LOADED = True
    Status.FILENAME = os.path.basename(filename)
    Status.EXTENSION = filename[filename.rfind('.')]

    print 'Feature Array:\n %s' % X
    print 'Target classifications:\n %s' % y

def load_file_gui():
    import wx

    def get_path(wildcard):
        app = wx.App(None)
        style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        dialog = wx.FileDialog(None, 'Open Files', wildcard=wildcard, style=style)
        if dialog.ShowModal() == wx.ID_OK:
            path = dialog.GetPath()
        else:
            path = None
        dialog.Destroy()
        app.Destroy()
        return path

    filename = get_path(("CSV files (*.csv)|*.csv|XLS files (*.xls)|*.xls|ARFF files (*.arff)|*.arff"))
    load_file(filename)

def load_random():
    """
    Generates a random dataset with 100 samples, 2 features, and 3 classes.
    """
    X, y, df = generate_random_points()
    loader = DatasetIO()
    loader.pickle_files([(X, 'load_X.pkl'), (y, 'load_y.pkl'), (df, 'df.pkl')],
                        Status.TEMP_DIR)

    # Update appropriate status flags.
    Status.DATASET_LOADED = True
    Status.FILENAME = 'RANDOM'

    print 'Feature Array:\n %s' % X
    print 'Target classifications:\n %s' % y

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


def visualize_dataset(command='', flags=(), plot_all=False, *args, **kwargs):
    """
    Create and display visualizations to user.
    """

    # Build parser for visualization
    parser = ArgumentParser()
    parser.add_argument('--suppress', action='store_true', dest='suppress',
                        help='Disable viewing of any generated plot(s).')
    p_args = parser.parse_args(flags)

    if Status.DATASET_LOADED:
        print "Creating visualization(s)",
        make_visualizations(command, plot_all)
        print ""
        if not p_args.suppress:
            print "Viewing generated plots..."
            view_saved_plots(command)
    else:
        raise InvalidCommandException("Can't visualize an unloaded dataset!")

def view_saved_plots(plot_name=''):
    # View all plots by default
    if plot_name == '':
        plot_name = '*.png'
        files = glob(join(Status.TEMP_DIR, plot_name))
    else:
        files = glob(join(Status.TEMP_DIR, plot_name + '.png'))

    for im_file in files:
        im = Image.open(im_file, 'r')
        im.show()

def see_images(*args):
    if '_temp/*.png' in args:
        files = glob('_temp/*.png')
    else:
        files = args

    for im_file in files:
        im = Image.open(im_file, 'r')
        im.show()

@progress(char='.', pause=0.5)
def make_visualizations(command='', plot_all=False):
    """
    Save the plots to _temp directory.
    """
    X, y, data_frame = get_pickled_dataset()
    class_name = data_frame.dtypes.index[-1]

    if command == 'plot_frequency' or plot_all:
        plot_class_frequency_bar(y)
    if command == 'plot_feature_matrix':
        plot_feature_matrix(data_frame)
    if command == 'plot_radial' or plot_all:
        plot_radial(data_frame, class_name)
    if command == 'plot_andrews' or plot_all:
        plot_andrews(data_frame, class_name)
    if command == 'plot_scatter_matrix':
        plot_scatter_matrix(data_frame)
    if command == 'plot_2d' or plot_all:
        plot_2d_dist(X, y)

def reset_plot_status():
    Status.FINISH_PLOTS = False

def plot_class_frequency_bar(target, bar_width=.35):
    plt.clf()
    # Get the frequency of each class label
    classes = np.unique(target)
    target_counts = Counter(target)

    # Plot the bar chart of class frequencies
    fig, ax = plt.subplots()
    ind = np.arange(len(classes))
    ax.set_xticks(ind)
    ax.bar(ind, target_counts.values(), width=bar_width, align='center')
    ax.set_title(Status.FILENAME)
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')
    ax.set_xticklabels(target_counts.keys())
    ax.set_title('Bar Chart of Class Label Frequencies')
    plt.savefig(join(Status.TEMP_DIR, Status.FREQ_NAME))


def plot_feature_matrix(data_frame):
    # Plot the matrix of feature-feature pairs
    plt.clf()
    g = sns.PairGrid(data_frame)
    g.map(plt.scatter)
    # plt.show(block=False)
    plt.title('Feature Matrix')
    plt.savefig(join(Status.TEMP_DIR, Status.FM_NAME))

def plot_radial(data_frame, class_name):
    plt.clf()
    radviz(data_frame, class_name)
    # plt.show(block=False)
    plt.title('Radial Plot')
    plt.savefig(join(Status.TEMP_DIR, Status.RADIAL_NAME))

def plot_andrews(data_frame, class_name):
    plt.clf()
    andrews_curves(data_frame, class_name)
    plt.title('Andrews Curve')
    # plt.show(block=False)
    plt.savefig(join(Status.TEMP_DIR, Status.ANDREWS_NAME))

def plot_scatter_matrix(data_frame):
    plt.clf()
    axes = scatter_matrix(data_frame, alpha=0.2, figsize=(10, 10), diagonal='kde')
    # plt.show(block=False)
    axes[0][0].set_title('Scatter Matrix with KDEs')
    plt.savefig(join(Status.TEMP_DIR, Status.SCM_NAME))

def plot_2d_dist(X, y):
    """
    Plots the feature array points on a plane.

    If the n_dims > 2, only consider the first two features.
    """
    plt.clf()
    from itertools import cycle
    colors = cycle('bgrcmyk')

    if len(X[0]) > 2:
        x_values = X[:, :2]
    else:
        x_values = X

    # Create a color-coded scatter plot by class label.
    for class_label, c in zip(np.unique(y), colors):
        xs = x_values[np.where(y == class_label)]
        plt.scatter(xs[:, 0], xs[:, 1], c=c, label=class_label)

    # Set plot labels and save.
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Distribution of Dataset ({})'.format(Status.FILENAME))
    plt.legend(loc='best')
    plt.savefig(join(Status.TEMP_DIR, Status.TD_NAME))

def dispatch_preprocess(args):
    if not Status.DATASET_LOADED:
        raise InvalidCommandException("Can't preprocess an unloaded dataset!")

    parser = ArgumentParser()
    parser.add_argument('-std', dest='std', action='store_true',
                        help='Standardize the feature array.')
    parser.add_argument('-norm', dest='norm', action='store_true',
                        help="Normalize the values of each feature.")
    p_args = parser.parse_args(args)
    pe = PreprocessingEngine()

    if p_args.std:
        print "Standardizing feature array..."
        X, y, _ = get_pickled_dataset()
        new_X = pe.standardize(X)
        print new_X
        update_feature_array(new_X)

    if p_args.norm:
        print "Normalizing feature array..."
        X, y, _ = get_pickled_dataset()
        new_X = pe.normalize_data(X)
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

    hist_file = os.path.join(os.environ['HOME'], '.pythonhistory')
    try:
        readline.read_history_file(hist_file)
    except IOError:
        pass

    # Set a limit on the number of commands to remember.
    # High values will hog system memory!
    readline.set_history_length(25)
    atexit.register(readline.write_history_file, hist_file)

    # Tab completion for GUI commands
    def completer(text, state):
        commands = Status.ALL_COMMANDS
        file_paths = []
        for dirname, dirnames, filenames in os.walk('.'):
            if '.git' in dirnames:
                # don't go into any .git directories.
                dirnames.remove('.git')
            # Add path to subdirectories
            file_paths.extend([os.path.join(dirname, sub_dir) for sub_dir in dirnames])
            # Add path to all filenames in subdirectories.
            file_paths.extend([os.path.join(dirname, filename) for filename in filenames])
            # Remove './' header in file strings.
            file_paths = [file.strip('./') for file in file_paths]

        options = [i for i in commands if i.startswith(text)]
        options.extend([f for f in file_paths if f.startswith(text)])

        try:
            return options[state]
        except IndexError:
            return None

    readline.set_completer(completer)

    # Bind tab completer to specific platforms
    if readline.__doc__ and 'libedit' in readline.__doc__:
        readline.parse_and_bind("bind -e")
        readline.parse_and_bind("bind '\t' rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    del hist_file, readline, rlcompleter, warnings


def quit_gui():
    shutil.rmtree(Status.TEMP_DIR)
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
    plot_2d                 Plot a 2-D distribution of the dataset.
    plot_andrews            Plots an Andrews curve of the dataset.

    plot_frequency          View the frequency of each class label.
    plot_feature_matrix     Generate a matrix plot of feature-feature
                            relationships.
    plot_scatter_matrix     Matrix plot with KDEs along the diagonal.
    plot_radial             Plot a radial chart of the dataset.
    preprocess [flags]      Preprocesses a dataset. Flags are
                                -std Standardize to mean 0 and variance 1
                                -norm Normalize each feature to range [0,1]
                                Eg. "preprocess -std"
    see_images [files]      View temporarily stored plots.
                                Eg. "see_images _temp/plot_2d.png"
    run                     Runs the ML alg on the loaded dataset.
        -A [alg]            REQUIRED flag! Options for [alg]:
                                dt = (Decision Tree)
        -test_ratio [0-1]   User can specify the test-train ratio.
        -cv [int]           Enables k-fold cross validation.
                            Example: "run -A dt -test_ratio .3 -cv 5"
    visualize               Plots all possible visualizations for input data.
        --suppress          Disable plotting output.
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
    elif command == 'load_file_gui':
        load_file_gui()
    elif command == 'preprocess':
        dispatch_preprocess(args)
    elif command in Status.PLOT_COMMANDS:
        visualize_dataset(command, args)
    elif command == 'visualize':
        visualize_dataset(flags=args, plot_all=True)
    elif command == 'see_images':
        see_images(*args)
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
        except AttributeError as ae:
            print ae.message
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
