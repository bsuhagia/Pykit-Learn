__author__ = 'Bhavesh'

from sklearn.linear_model import LinearRegression, LogisticRegression, RANSACRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from numpy import inf

def train_leastSquareModel(X, y, fit_intercept=True, normalize=False,
                           copy_X=True, n_jobs=1):
    """
    Train a regression model using Least Square method
    """
    model = LinearRegression(fit_intercept=fit_intercept,
                                          normalize=normalize,
                                          copy_X=copy_X,
                                          n_jobs=n_jobs)
    model = model.fit(X, y)
    return model

def train_kernelRidgeModel(X, y, alpha=1, kernel='linear',gamma=None, degree=3,
                      coef0=1, kernel_params=None):
    """
    Train a kernel ridge regression model
    """
    model = KernelRidge(alpha=alpha,
                        kernel=kernel,
                        gamma=gamma,
                        degree=degree,
                        coef0=coef0,
                        kernel_params=kernel_params)
    model = model.fit(X, y)
    return model

def train_logisticRegressionModel(X, y, penalty='l2', dual=False, tol=0.0001,
                                  C=1.0, fit_intercept=True, intercept_scaling=1,
                                  class_weight=None, random_state=None,
                                  solver='liblinear', max_iter=100,
                                  multi_class='ovr', verbose=0):
    """
    Train a logistic regression model
    """
    model = LogisticRegression(penalty=penalty,
                               dual=dual,
                               tol=tol,
                               C=C,
                               fit_intercept=fit_intercept,
                               intercept_scaling=intercept_scaling,
                               class_weight=class_weight,
                               random_state=random_state,
                               solver=solver,
                               max_iter=max_iter,
                               multi_class=multi_class,
                               verbose=verbose)
    model = model.fit(X,y)
    return model

def train_polynomialRegressionModel(X, y, degree=2, interaction_only=False,
                                    include_bias=True):
    """
    Train a polynomial model using Linear Regression Pipeline with degrees
    """
    model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                      ('linear', LinearRegression(fit_intercept=False))])
    model = model.fit(X, y)
    return model

def train_BayesianRegressionModel(X, y,n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
    """
    Train a Bayesian regression model
    """
    model = BayesianRidge(n_iter=n_iter,
                          tol=tol,
                          alpha_1=alpha_1,
                          alpha_2=alpha_2,
                          lambda_1=lambda_1,
                          lambda_2=lambda_2,
                          compute_score=compute_score,
                          fit_intercept=fit_intercept,
                          normalize=normalize,
                          copy_X=copy_X,
                          verbose=verbose)
    model = model.fit(X,y)
    return model

def train_RANSACRegressionModel(X, y, base_estimator=None, min_samples=None, residual_threshold=None, is_data_valid=None, is_model_valid=None, max_trials=100, stop_n_inliers=inf, stop_score=inf, stop_probability=0.99, residual_metric=None, random_state=None):
    """
    Train a RANSAC regression model
    """
    model = RANSACRegressor(base_estimator=base_estimator,
                            min_samples=min_samples,
                            residual_threshold=residual_threshold,
                            is_data_valid=is_data_valid,
                            is_model_valid=is_model_valid,
                            max_trials=max_trials,
                            stop_n_inliers=stop_n_inliers,
                            stop_score=stop_score,
                            stop_probability=stop_probability,
                            residual_metric=residual_metric,
                            random_state=random_state)
    model = model.fit(X, y)
    return model




