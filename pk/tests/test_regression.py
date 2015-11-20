"""
__author__ = 'Bhavesh'


from pk.utils.loading import *
from pk.utils.regression_utils import *
from pk.utils.metrics import *
from prettytable import PrettyTable
def runall_regression(X, y):
    T = PrettyTable(["Regression Method", "Train Accuracy (%)", "Test Accuracy (%)", "Variance score", "Mean Squared Error", "Mean Abs Error", "Median Abs Error", "R2 score"])
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3,random_state=0)

    # leastsquare
    ls = train_leastSquareModel(X_train, y_train)
    ls_train_acc = get_train_accuracy(ls, X_train, y_train)
    ls_test_acc = get_test_accuracy(ls, X_test, y_test)
    ls_var = get_variance_score(ls, X_test, y_test)
    ls_mse = get_mean_squared_error(ls, X_test, y_test)
    ls_mean = get_mean_abs_error(ls, X_test, y_test)
    ls_med = get_median_abs_error(ls, X_test, y_test)
    ls_r2 = get_r2_score(ls, X_test, y_test)
    T.add_row((["Least Square Linear", ls_train_acc, ls_test_acc, ls_var, ls_mse, ls_mean, ls_med, ls_r2]))

    # polynomial model with degree 3
    poly = train_polynomialRegressionModel(X_train, y_train, degree=3)
    poly_train_acc = get_train_accuracy(poly, X_train, y_train)
    poly_test_acc = get_test_accuracy(poly, X_test, y_test)
    poly_var = get_variance_score(poly, X_test, y_test)
    poly_mse = get_mean_squared_error(poly, X_test, y_test)
    poly_mean = get_mean_abs_error(poly, X_test, y_test)
    poly_med = get_median_abs_error(poly, X_test, y_test)
    poly_r2 = get_r2_score(poly, X_test, y_test)
    T.add_row((["Polynomial (degree = 3)", poly_train_acc, poly_test_acc, poly_var, poly_mse, poly_mean, poly_med, poly_r2]))

    # logistic regression
    log = train_logisticRegressionModel(X_train, y_train)
    log_var = get_variance_score(log, X_test, y_test)
    log_mse = get_mean_squared_error(log, X_test, y_test)
    log_mean = get_mean_abs_error(log, X_test, y_test)
    log_med = get_median_abs_error(log, X_test, y_test)
    log_r2 = get_r2_score(log, X_test, y_test)
    T.add_row((["Logistic", "NA", "NA", log_var, log_mse, log_mean, log_med, log_r2]))

    # RANSAN
    ransac = train_RANSACRegressionModel(X_train, y_train)
    ransac_train_acc = get_train_accuracy(ransac, X_train, y_train)
    ransac_test_acc = get_test_accuracy(ransac, X_test, y_test)
    ransac_var = get_variance_score(ransac, X_test, y_test)
    ransac_mse = get_mean_squared_error(ransac, X_test, y_test)
    ransac_mean = get_mean_abs_error(ransac, X_test, y_test)
    ransac_med = get_median_abs_error(ransac, X_test, y_test)
    ransac_r2 = get_r2_score(ransac, X_test, y_test)
    T.add_row((["RANSAC", ransac_train_acc, ransac_test_acc, ransac_var, ransac_mse, ransac_mean, ransac_med, ransac_r2]))

    # Bayes
    bayes = train_BayesianRegressionModel(X_train, y_train)
    bayes_train_acc = get_train_accuracy(bayes, X_train, y_train)
    bayes_test_acc = get_test_accuracy(bayes, X_test, y_test)
    bayes_var = get_variance_score(bayes, X_test, y_test)
    bayes_mse = get_mean_squared_error(bayes, X_test, y_test)
    bayes_mean = get_mean_abs_error(bayes, X_test, y_test)
    bayes_med = get_median_abs_error(bayes, X_test, y_test)
    bayes_r2 = get_r2_score(bayes, X_test, y_test)
    T.add_row((["Bayesian", bayes_train_acc, bayes_test_acc, bayes_var, bayes_mse, bayes_mean, bayes_med, bayes_r2]))

    # Kernel ridge
    kr = train_kernelRidgeModel(X_train, y_train)
    kr_train_acc = get_train_accuracy(kr, X_train, y_train)
    kr_test_acc = get_test_accuracy(kr, X_test, y_test)
    kr_var = get_variance_score(kr, X_test, y_test)
    kr_mse = get_mean_squared_error(kr, X_test, y_test)
    kr_mean = get_mean_abs_error(kr, X_test, y_test)
    kr_med = get_median_abs_error(kr, X_test, y_test)
    kr_r2 = get_r2_score(kr, X_test, y_test)
    T.add_row((["Kernel Ridge", kr_train_acc, kr_test_acc, kr_var, kr_mse, kr_mean, kr_med, kr_r2]))
    print T

# dataset for regression
X, y, _ = load_csv('concrete.csv')
runall_regression(X, y)

"""
