__author__ = 'Bhavesh'

from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import SGDClassifier

from prygress import *

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
    print 'Decision Tree done!'
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
    print 'SVM completed!'
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
    print 'KNN completed!'
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
    print 'Naive Bayes completed!'
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
    print 'AdaBoost completed!'
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
    print 'Linear Discriminant Analysis completed!'
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
    print 'Quadratic Discriminant Analysis completed!'
    return clf

def train_bagging(X, y, base_estimator=None, n_estimators=10, max_samples=1.0,
                  max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, n_jobs=1,
                  random_state=None, verbose=0):
    """
    Builds a Bagging model based on decision tree

    Returns:
    clf: Fitted Bagging classifier
    """
    clf = BaggingClassifier(base_estimator=base_estimator,
                            n_estimators=n_estimators,
                            max_samples=max_samples,
                            max_features=max_features,
                            bootstrap=bootstrap,
                            bootstrap_features=bootstrap_features,
                            oob_score=oob_score,
                            n_jobs=n_jobs,
                            random_state=random_state,
                            verbose=verbose)
    clf = clf.fit(X,y)
    return clf

def train_randomForest(X, y, n_estimators=10, criterion='gini', max_depth=None,
                       min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                       max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False,
                       n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None):
    """

    Builds a random forest classifier

    Returns: Fitted random forest model
    """
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 criterion=criterion,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                 max_features=max_features,
                                 max_leaf_nodes=max_leaf_nodes,
                                 bootstrap=bootstrap,
                                 oob_score=oob_score,
                                 n_jobs=n_jobs,
                                 random_state=random_state,
                                 verbose=verbose,
                                 warm_start=warm_start,
                                 class_weight=class_weight)
    clf = clf.fit(X,y)
    return clf

def train_stochaticGradientDescent(X, y, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15,
                                   fit_intercept=True, n_iter=5, shuffle=True, verbose=0,
                                   epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal',
                                   eta0=0.0, power_t=0.5, class_weight=None, warm_start=False,
                                   average=False):
    clf = SGDClassifier(loss=loss,
                        penalty=penalty,
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        fit_intercept=fit_intercept,
                        n_iter=n_iter,
                        shuffle=shuffle,
                        verbose=verbose,
                        epsilon=epsilon,
                        n_jobs=n_jobs,
                        random_state=random_state,
                        learning_rate=learning_rate,
                        eta0=eta0,
                        power_t=power_t,
                        class_weight=class_weight,
                        warm_start=warm_start,
                        average=average
                        )
    clf = clf.fit(X,y)
    return clf