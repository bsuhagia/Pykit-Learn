# __author__ = 'Bhavesh'
#
from sklearn.metrics import confusion_matrix, explained_variance_score, mean_squared_error, mean_absolute_error, r2_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, silhouette_score, v_measure_score
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
    avg = scores.mean()
    return scores, round(avg*100, 5)

# def get_variance_score(clf, X_test, true_y):
#     pred_y = clf.predict(X_test)
#     return round(explained_variance_score(true_y, pred_y), 4)
#
# def get_mean_abs_error(clf, X_test, true_y):
#     pred_y = clf.predict(X_test)
#     return round(mean_absolute_error(true_y, pred_y), 4)
#
# def get_mean_squared_error(clf, X_test, true_y):
#     pred_y = clf.predict(X_test)
#     return round(mean_squared_error(true_y, pred_y), 4)
#
# def get_median_abs_error(clf, X_test, true_y):
#     pred_y = clf.predict(X_test)
#     return round(median_absolute_error(true_y, pred_y), 4)
#
# def get_r2_score(clf, X_test, true_y):
#     pred_y = clf.predict(X_test)
#     return round(r2_score(true_y, pred_y), 4)
#
# def get_adjusted_rand_index(clf, X_test, true_y):
#     pred_y = clf.predict(X_test)
#     return round(adjusted_rand_score(true_y, pred_y), 4)
#
# def get_adjusted_mutual_info(clf, X_test, true_y):
#     pred_y = clf.predict(X_test)
#     return round(adjusted_mutual_info_score(true_y, pred_y), 4)
#
# def get_homogeneity_score(clf, X_test, true_y):
#     pred_y = clf.predict(X_test)
#     return round(homogeneity_score(true_y, pred_y), 4)
#
# def get_vscore(clf, X_test, true_y):
#     pred_y = clf.predict(X_test)
#     return round(v_measure_score(true_y, pred_y), 4)
#
# def get_silhouette_score(clf, X):
#     # pred_y = clf.predict(X_test)
#     return round(silhouette_score(X, clf.means_, metric='euclidean'))
#
# def benchmark(X, y, training_func, *args, **kwargs):
#     clf = training_func(X, y, *args, **kwargs)
#     get_train_accuracy(clf, X, y)
#     get_test_accuracy(clf, X, y)
