__author__ = 'Bhavesh'

from sklearn.metrics import confusion_matrix, explained_variance_score, mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation

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
    return round(((clf.score(X, y))*100),5)


def get_test_accuracy(clf, X, y):
    return round(((clf.score(X, y))*100),5)


def get_cv_accuracy(clf, X, y, cv=10):
    scores = cross_validation.cross_val_score(clf, X, y, cv=cv)
    # print 'Scores: ' + ', '.join(map(str, scores))
    avg = scores.mean()
    # print 'Average accuracy: %f (+/- %f)' % (avg, scores.std() * 2)
    return scores, round(avg*100, 5)

def get_variance_score(clf, X_test, true_y):
    pred_y = clf.predict(X_test)
    return round(explained_variance_score(true_y, pred_y), 4)

def get_mean_abs_error(clf, X_test, true_y):
    pred_y = clf.predict(X_test)
    return round(mean_absolute_error(true_y, pred_y), 4)

def get_mean_squared_error(clf, X_test, true_y):
    pred_y = clf.predict(X_test)
    return round(mean_squared_error(true_y, pred_y), 4)

def get_median_abs_error(clf, X_test, true_y):
    pred_y = clf.predict(X_test)
    return round(median_absolute_error(true_y, pred_y), 4)

def get_r2_score(clf, X_test, true_y):
    pred_y = clf.predict(X_test)
    return round(r2_score(true_y, pred_y), 4)

def benchmark(X, y, training_func, *args, **kwargs):
    clf = training_func(X, y, *args, **kwargs)
    get_train_accuracy(clf, X, y)
    get_test_accuracy(clf, X, y)
