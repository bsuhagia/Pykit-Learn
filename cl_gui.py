# Author: Sean Dai

import shutil
import sys
import os
import cPickle
import numpy as np

from optparse import OptionParser
from collections import Counter
from pk.utils.loading import *
from pk.utils.preprocess_utils import *

class Status(object):
    DATASET_LOADED = False

class InvalidCommandException(Exception):
    def __init__(self, message, errors=[]):
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
    X, y = _load_file(filename)
    pickle_files(*[(X, 'load_X.pkl'), (y, 'load_y.pkl')])
    Status.DATASET_LOADED = True
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

def visualize_dataset():
    if Status.DATASET_LOADED:
        X, y = cPickle.load(open('_temp/load_X.pkl', 'r')), cPickle.load(open('_temp/load_y.pkl', 'r'))
        classes = np.unique(y)
        target_counts = Counter(y)
        import matplotlib.pyplot as plt
        plt.bar(np.arange(classes), y)
        plt.xticks(classes)
        plt.show()
    else:
        raise Exception("Can't visualize an unloaded dataset!")

def process(line):
    tokens = tuple(line.split(' '))
    command, args = tokens[0], tokens[1:]
    if command == 'load':
        load_file(*args)
    elif command == 'visualize_dataset':
        print "Visualizing..."
        visualize_dataset()
    else:
        raise InvalidCommandException("{} is not a recognized command.".format(command))

def setup():
    if not os.path.exists("_temp/"):
        os.mkdir("_temp/")

def main():
    """
    To run, type "python cl_gui.py"
    """
    print "Welcome to the command-line version of Pykit-Learn!"
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
            print e.message
        except KeyboardInterrupt:
            shutil.rmtree("_temp")
            sys.exit(1)
if __name__ == "__main__":
    main()