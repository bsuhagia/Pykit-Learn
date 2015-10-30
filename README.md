# Scikit-learn GUI
A simple GUI for doing fast-paced machine learning using Python

To run all tests, install the **nose** Python module, and enter the following command in the root directory of the project:
```
nosetests
```

## Todo List
- [ ] Load dataset files (categorical/numeric data)
  - [x] .arff
  - [x] .csv
  - [x] .xls/.xlsx
- [ ] Preprocessing data
  - [x] Standardization
  - [x] Normalization of training examples
  - [ ] Feature Binarization
  - [x] Remove examples with '?' missing values
- [ ] Supervised Learning
  - [ ] Linear & Quadratic Discriminant Analysis
  - [ ] SVMs
  - [ ] Stochastic Gradient Descent
  - [ ] kNN
  - [ ] Decision Trees
  - [ ] Ensemble Methods
    - [ ] Bagging
    - [ ] Randomized Trees
    - [ ] AdaBoost
  - [ ] Multiclass and Multilabel Algorithms
  - [ ] Feature Selection
    - [ ] Variance thresholding
    - [ ] Univariate feature selection
  - [ ] Generalized Linear Models
    - [ ] Least Squares
    - [ ] RANSAC
    - [ ] Bayesian
    - [ ] Logistic
    - [ ] Polynomial
  - [ ] Kernel Ridge Regression
- [ ] Unsupervised Learning
  - [ ] Gaussian Mixture Models
    - [ ] GMM
    - [ ] DPGMM
  - [ ] Manifold Learning
  - [ ] Clustering
    - [ ] K-means
    - [ ] Spectral clustering
    - [ ] Hierarchical clustering
    - [ ] DBSCAN
  - [ ] Decomposing signals into components
    - [ ] PCA
    - [ ] ICA
    - [ ] Factor Analysis
  - [ ] Covariance Estimation
  - [ ] Novelty and Outlier Detection
  - [ ] Restricted Boltzmann Machines
- [ ] Model Selection and Evaluation
  - [ ] Cross Validation
  - [ ] Grid Search
  - [ ] Prediction Metrics
    - [ ] Classification Metrics
      - [ ] ROC
      - [ ] Accuracy Score
      - [ ] Confusion Matrix
    - [ ] Regression Metrics
      - [ ] MAE, MSE, R2
    - [ ] Clustering Metrics
      - [ ] Adjusted Rand index
      - [ ] Homogeneity (similarity of items within cluster)
      - [ ] Completeness (same class items all go in one cluster)
  - [ ] Validation Curves
- [ ] Dataset Transformations
  - [ ] Pipelining
  - [ ] Feature Extraction
    - [x] Dictionary Vectorization
  - [ ] Kernel Approximation
- [ ] Dataset Loading Utilities
  - [ ] Download data from mldata.org
  - [x] Generate a random dataset w/ class labels
- [ ] Visualizations
    - [x] Image segmentation demo