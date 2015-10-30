__author__ = 'Bhavesh'

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA

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

def train_svm(X, y, C=1.0, kernel='linear', degree=3, gamma=0.0, coef0=0.0,
              shrinking=True, probability=False, tol=0.001, cache_size=200,
              class_weight=None, verbose=False, max_iter=-1, random_state=None):
    """
    Builds a support vector machine model

    Returns:
    clf: Fitted SVM classifier object
    """
    clf = svm.SVC(C=C,
                  kernel=kernel,
                  degree=degree,
                  gamma=gamma,
                  coef0=gamma,
                  shrinking=shrinking,
                  probability=probability,
                  tol=tol,
                  cache_size=cache_size,
                  class_weight=class_weight,
                  verbose=verbose,
                  max_iter=max_iter,
                  random_state=random_state)
    clf = clf.fit(X, y)
    return clf

def train_knn(X, y, n_neighbors=5, weights='uniform', algorithm='auto',
              leaf_size=30, p=2, metric='minkowski', metric_params=None):
    """
    Builds a k-nearest neighbor model

    Returns:
    clf: Fitted nearest neighbor model
    """
    clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                               weights=weights,
                               algorithm=algorithm,
                               leaf_size=leaf_size,
                               p=p,
                               metric=metric,
                               metric_params=metric_params)
    clf = clf.fit(X, y)
    return clf

def train_naive_bayes(X, y, distribution='Gaussian'):
    """
    Builds a naive bayes classification model

    Returns:
    clf: Fitted naive bayes model
    """
    if (distribution == 'Guassian'):
        clf = GaussianNB()
    elif (distribution == 'Multinomial'):
        clf = MultinomialNB()
    else:
        clf = BernoulliNB()
    clf = clf.fit(X,y)
    return clf

def train_adaboost(X, y, base_estimator=DecisionTreeClassifier, n_estimators=50, learning_rate=1.0,
                   algorithm='SAMME.R', random_state=None):
    """
    Builds a Boost classifier with decision tree as base estimator

    Returns:
    clf: Fitted ada boost model
    """
    clf = AdaBoostClassifier(base_estimator=base_estimator,
                             n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             algorithm=algorithm,
                             random_state=random_state)
    clf = clf.fit(X,y)
    return clf

def train_lda(X, y, solver='svd', shrinkage=None, priors=None, n_components=None,
              store_covariance=False, tol=0.0001):
    """
    Builds a linear discriminant analysis model

    Returns:
    clf: Fitted LDA model
    """
    clf  = LDA(solver=solver,
               shrinkage=shrinkage,
               priors=priors,
               n_components=n_components,
               store_covariance=store_covariance,
               tol=tol)
    clf = clf.fit(X,y)
    return clf

def train_qda(X, y, priors=None, reg_param=0.0):
    """
    Builds a quadratic discriminant analysis model

    Returns:
    clf: Fitted QDA model
    """
    clf = QDA(priors=priors,
              reg_param=reg_param)
    clf = clf.fit(X,y)
    return clf