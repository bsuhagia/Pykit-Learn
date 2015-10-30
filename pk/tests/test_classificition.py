__author__ = 'Bhavesh'
from pk.utils.performance_utils import *
from pk.utils.classification_utils import *
from sklearn import cross_validation

def runAll(X, y):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3,random_state=0)
    dts = train_decision_tree(X_train, y_train)
    knn = train_knn(X_train, y_train)
    svm = train_svm(X_train, y_train)
    nb = train_naive_bayes(X_train, y_train)
    ada = train_adaboost(X_train, y_train)
    lda = train_lda(X_train, y_train)
    qda = train_qda(X_train, y_train)


