__author__ = 'Bhavesh'

from pk.utils.performance_utils import *
from pk.utils.classification_utils import *
from prettytable import PrettyTable
from pk.utils.loading import *
import warnings

def runAll(X, y):
    warnings.filterwarnings('ignore')
    T = PrettyTable(["Method", "Train Accuracy (%)", "Test Accuracy (%)", "Cross Validation Accuracy (%)"])
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3,random_state=0)

    dts = train_decision_tree(X_train, y_train)
    dts_train_acc = get_train_accuracy(dts, X_train, y_train)
    dts_test_acc = get_test_accuracy(dts, X_test, y_test)
    _, dts_cv_acc = get_cv_accuracy(dts, X, y)
    T.add_row((["Decision Tree", dts_train_acc, dts_test_acc, dts_cv_acc]))

    knn = train_knn(X_train, y_train)
    knn_train_acc = get_train_accuracy(knn, X_train, y_train)
    knn_test_acc = get_test_accuracy(knn, X_test, y_test)
    _, knn_cv_acc = get_cv_accuracy(dts, X, y)
    T.add_row((["Nearest Neighbor", knn_train_acc, knn_test_acc, knn_cv_acc]))

    svm = train_svm(X_train, y_train)
    svm_train_acc = get_train_accuracy(svm, X_train, y_train)
    svm_test_acc = get_test_accuracy(svm, X_test, y_test)
    _, svm_cv_acc = get_cv_accuracy(svm, X, y)
    T.add_row((["Support Vector Machine", svm_train_acc, svm_test_acc, svm_cv_acc]))

    nb = train_naive_bayes(X_train, y_train)
    nb_train_acc = get_train_accuracy(nb, X_train, y_train)
    nb_test_acc = get_test_accuracy(nb, X_test, y_test)
    _, nb_cv_acc = get_cv_accuracy(nb, X, y)
    T.add_row((["Naive Bayes", nb_train_acc, nb_test_acc, nb_cv_acc]))

    ada = train_adaboost(X_train, y_train, base_estimator=dts)
    ada_train_acc = get_train_accuracy(ada, X_train, y_train)
    ada_test_acc = get_test_accuracy(ada, X_test, y_test)
    _, ada_cv_acc = get_cv_accuracy(ada, X, y)
    T.add_row((["AdaBoost", ada_train_acc, ada_test_acc, ada_cv_acc]))

    lda = train_lda(X_train, y_train)
    lda_train_acc = get_train_accuracy(lda, X_train, y_train)
    lda_test_acc = get_test_accuracy(lda, X_test, y_test)
    _, lda_cv_acc = get_cv_accuracy(lda, X, y)
    T.add_row((["Linear Discriminant Analysis", lda_train_acc, lda_test_acc, lda_cv_acc]))

    qda = train_qda(X_train, y_train)
    qda_train_acc = get_train_accuracy(qda, X_train, y_train)
    qda_test_acc = get_test_accuracy(qda, X_test, y_test)
    _, qda_cv_acc = get_cv_accuracy(qda, X, y)
    T.add_row((["Quadratic Discriminant Analysis", qda_train_acc, qda_test_acc, qda_cv_acc]))

    bag = train_bagging(X_train, y_train, base_estimator=dts)
    bag_train_acc = get_train_accuracy(bag, X_train, y_train)
    bag_test_acc = get_test_accuracy(bag, X_test, y_test)
    _, bag_cv_acc = get_cv_accuracy(bag, X, y)
    T.add_row((["Bagging", bag_train_acc, bag_test_acc, bag_cv_acc]))

    rf = train_randomForest(X_train, y_train)
    rf_train_acc = get_train_accuracy(rf, X_train, y_train)
    rf_test_acc = get_test_accuracy(rf, X_test, y_test)
    _, rf_cv_acc = get_cv_accuracy(rf, X, y)
    T.add_row((["Random Forest", rf_train_acc, rf_test_acc, rf_cv_acc]))

    sgd = train_stochaticGradientDescent(X_train, y_train)
    sgd_train_acc = get_train_accuracy(sgd, X_train, y_train)
    sgd_test_acc = get_test_accuracy(sgd, X_test, y_test)
    _, sgd_cv_acc = get_cv_accuracy(sgd, X, y)
    T.add_row((["Stochastic Gradient Descent", sgd_train_acc, sgd_test_acc, sgd_cv_acc]))
    print T

X_data, y_data, dataset = load_csv('iris2.csv')
runAll(X_data,y_data)
