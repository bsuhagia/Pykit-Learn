Scikit-learn GUI
================
A simple GUI for doing fast-paced machine learning in Python.

Release Notes
-------------
### Version 0.1
  

Installation Guide
------------------
The installation instructions are written primarily for Mac OSX machines. (although I think they might work on Linux systems too)

### Mac OSX
  
In order to run the project, install the following dependencies:

1. [Python](https://drive.google.com/open?id=1TYYzeYfbz6GQZPwTKHsBl528ujDL7akzCeEhxbaLMls)

2. [pip](https://pip.pypa.io/en/stable/installing/)

3. [scikit-Learn](http://scikit-learn.org/stable/install.html)

4. [numpy](https://drive.google.com/open?id=1TYYzeYfbz6GQZPwTKHsBl528ujDL7akzCeEhxbaLMls)

5. [scipy](https://drive.google.com/open?id=1TYYzeYfbz6GQZPwTKHsBl528ujDL7akzCeEhxbaLMls)

6. [matplotlb](http://matplotlib.org/users/installing.html)

7. [seaborn](http://stanford.edu/~mwaskom/software/seaborn/installing.html)

8. [PIL](http://www.pythonware.com/products/pil/)

9. [pandas](http://pandas.pydata.org/pandas-docs/stable/install.html)

10. [nose](https://nose.readthedocs.org/en/latest/)

11. [PyQt](http://pyqt.sourceforge.net/Docs/PyQt4/installation.html)

Alternatively, you can enter the makefile command
```
make install
```
to automatically install dependencies 3-10.


## User Manual
To start the application, enter the command at the root directory of the project:

```
python cl_gui.py
```
A list of commands for the Pykit-Learn application is provided in the table below: 

Commands                     | Example                               | Description 
---------------------------- | ------------------------------------- | ----------------
**load** [file]              | `load ~/Downloads/data.csv`           | Loads the dataset at the path specified by [file]. No quotes "" around the filename!
**load_random**              | `load_random`                         | Load a randomly generated dataset with 3 classes.
**plot_2d**                  |                                       | Plot a 2-D distribution of the dataset.
**plot_andrews**             |                                       | Plot an Andrews curve of the dataset.
**plot_frequency**           |                                       | View the frequency of each class label.
**plot_feature_matrix**      |                                       | Generate a matrix plot of feature-feature relationships.
**plot_scatter_matrix**      |                                       | Matrix plot with KDEs along the diagonal.
**plot_radial**              |                                       | Plot a radial chart of the dataset.
**preprocess** [flags]       | `preprocess -std`                     | Preprocesses a dataset. Flags: **-std** Standardize to mean 0 and variance 1. **-norm** Normalize each feature to range [0,1]
**run** -A [alg] -test_ratio [0-1] -cv [int] | `run -A dt -test_ratio .3 -cv 5` | Runs the ML alg on the loaded dataset. **alg** = dt (Decision Tree). Can specify the test-train ratio. **-cv** enables k-fold cross validation.
**visualize** --suppress     |                                       | Plots all possible visualizations for input data. **--suppress** disables all plotting output.
**help**                     |                                       | Provides a help screen of available commands.
**quit**                     |                                       | Quits the command line GUI.


## Examples

## Testing
The unit-testing framework used in this project is the **nose** Python module. Running the unit tests yourself
is as simple as entering the following command in the root directory of the project:
```
make test
```
To run all the unit tests (this might take some time), type
```
make test-all
```

## Todo List
- [x] MVC Components
    - [x] Model Classes
        - [x] Algorithm
        - [x] SupervisedAlgorithm
        - [x] UnsupervisedAlgorithm
        - [x] RegressionAlgorithm
        - [x] ExecutionReport
    - [x] Controller Classes
        - [x] AlgorithmEngine
        - [x] DatasetIO
        - [x] PreprocessingEngine
        - [x] Visualizer
    - [x] View Classes
        - [x] BaseView
- [x] Demos
  - [x] Image segmentation demo
  - [x] Command-line GUI
- [x] Loading
    - [x] File formats
      - [x] .arff
      - [x] .csv
      - [x] .xls/.xlsx
  - [x] Generate random Gaussian data w/ labels
  - [x] Download dataset from mldata.org
- [x] Preprocessing data
  - [x] Standardization
  - [x] Normalization of training examples
  - [x] Feature Binarization
  - [x] Remove examples with '?' missing values
  - [x] Imputation of missing values
  - [x] Numerical encoding of categorical features
- [x] Supervised Learning
  - [x] Linear & Quadratic Discriminant Analysis
  - [x] SVMs
  - [x] Stochastic Gradient Descent
  - [x] kNN
  - [x] Decision Trees
  - [x] Ensemble Methods
    - [x] Bagging
    - [x] Randomized Trees
    - [x] AdaBoost
  - [x] Multiclass and Multilabel Algorithms
  - [x] Feature Selection
    - [x] Variance thresholding
    - [x] Univariate feature selection
  - [x] Generalized Linear Models
    - [x] Least Squares
    - [x] RANSAC
    - [x] Bayesian
    - [x] Logistic
    - [x] Polynomial
  - [x] Kernel Ridge Regression
- [x] Unsupervised Learning
  - [x] Gaussian Mixture Models
    - [x] GMM
    - [x] DPGMM
  - [x] Manifold Learning
  - [x] Clustering
    - [x] K-means
    - [x] Spectral clustering
    - [x] Hierarchical clustering
    - [x] DBSCAN
  - [x] Decomposing signals into components
    - [x] PCA
    - [x] ICA
    - [x] Factor Analysis
  - [x] Covariance Estimation
  - [x] Novelty and Outlier Detection
  - [x] Restricted Boltzmann Machines
- [x] Model Selection and Evaluation
  - [x] Cross Validation
  - [x] Grid Search
  - [x] Prediction Metrics
    - [x] Classification Metrics
      - [x] ROC
      - [x] Accuracy Score
      - [x] Confusion Matrix
    - [x] Regression Metrics
      - [x] MAE, MSE, R2
    - [x] Clustering Metrics
      - [x] Adjusted Rand index
      - [x] Homogeneity (similarity of items within cluster)
      - [x] Completeness (same class items all go in one cluster)
  - [x] Validation Curves
- [x] Dataset Transformations
  - [x] Pipelining
  - [x] Feature Extraction
    - [x] Dictionary Vectorization
  - [x] Kernel Approximation
- [x] Visualizations
    - [x] Plotting features (2d, frequency chart, radial plot, etc.)



