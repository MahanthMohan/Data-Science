import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(0)

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2,-1], [2,-3], [1,1]], cluster_std=0.9)

plt.scatter(X[:,0], X[:,1], marker='.')
plt.show()

k_means = KMeans(init = 'k-means++', n_clusters=4, n_init=12)
k_means.fit(X)

k_means_labels = k_means.labels_
print("The labels for the blobs: " + str(k_means_labels))

k_means_cluster_centers = k_means.cluster_centers_
print("The K-centers are: ")
print(k_means_cluster_centers)
        
