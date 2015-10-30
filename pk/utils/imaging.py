import time

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import imread
from sklearn.cluster import DBSCAN


def segment_image(im_file, n_segments=5, alg='ac'):
    img = imread(im_file)
    img = img[:,:,0]
    X = np.reshape(img, (-1, 1))

    if alg == 'ac':
        # Define the structure A of the data. Pixels connected to their neighbors.
        connectivity = grid_to_graph(*img.shape)

        # Compute clustering
        print("Compute structured hierarchical clustering...")
        st = time.time()
        n_clusters = n_segments  # number of regions
        ward = AgglomerativeClustering(n_clusters=n_clusters,
                linkage='complete', connectivity=connectivity).fit(X)
        label = np.reshape(ward.labels_, img.shape)
    elif alg == 'dbscan':
        print("Compute DBScan clustering...")
        st = time.time()
        dbs = DBSCAN(eps=1).fit(X)
        label = np.reshape(dbs.labels_, img.shape)

    print("Elapsed time: ", time.time() - st)
    print("Number of pixels: ", label.size)
    print("Number of clusters: ", np.unique(label).size)

    return label

def visualize_segments(label, type='mask', im_file=None):
    if type == 'mask':
        plt.imshow(label, cmap=plt.cm.Paired)
    elif type == 'contour':
        if im_file is not None:
            img = imread(im_file)
            n_clusters = np.unique(label).size
            plt.imshow(img, cmap=plt.cm.gray)
            for cluster_i in range(n_clusters):
                plt.contour(label == cluster_i, contours=1,
                            colors=[plt.cm.spectral(
                                cluster_i/float(n_clusters)),])
    plt.xticks(())
    plt.yticks(())
    plt.show()

# im_file = "/Users/sd/Downloads/sample_images/biking.jpg"
# segment_labels = segment_image(im_file, n_segments=15, alg='ac')
# visualize_segments(segment_labels, type='contour', im_file=im_file)
# visualize_segments(segment_labels, type='mask', im_file=im_file)