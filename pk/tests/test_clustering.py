__author__ = 'Bhavesh'
from pk.utils.clustering_utils import *
from pk.utils.loading import *
from pk.utils.performance_utils import *
from prettytable import PrettyTable



def runall_clustering(X, n_clusters, y=None):
    T = PrettyTable(["Clustering Method", "Silhoutte Scores"])
    gmm = train_gmm(X,n_components=n_clusters)
    gmm_sil = get_silhouette_score(gmm, X)
    T.add_row((["Gaussian Mixture Models", gmm_sil]))
    print T

X, y, _ = load_csv('iris.csv')
runall_clustering(X, 3)


    # dpgmm

    # kmeans

    # spectral

    # dbscan

