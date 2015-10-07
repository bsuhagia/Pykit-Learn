# Author: Sean Dai
import numpy as np
from scipy.io.arff import loadarff

def load_arff(filename):
    """
    Loads .arff dataset files.

    Args:
        filename: str

    Returns:
        X : a (num_examples x num_features) numpy array of examples X
        y : the class labels y of size (1, num_examples)
    """
    dataset = loadarff(open(filename,'r'))
    features = dataset[1].names()
    class_attr = features[-1]
    y = np.array(dataset[0][class_attr])
    X = np.array(dataset[0][features[:-1]])
    X = np.array(map(list, X))
    return X,y