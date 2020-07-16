import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(0)

centers = [[4,4], [-2,-1], [2,-3], [1,1]]
X, y = make_blobs(n_samples=5000, centers=centers, cluster_std=0.9)

k_means = KMeans(init = 'k-means++', n_clusters=4, n_init=12)
k_means.fit(X)

k_means_labels = k_means.labels_
print("The labels for the blobs: " + str(k_means_labels))

k_means_cluster_centers = k_means.cluster_centers_
print("The K-centers are: ")
print(k_means_cluster_centers)

fig = plt.figure()
ax = fig.add_subplot()
colors = plt.cm.Spectral(np.linspace(0, 1, len(k_means_labels)))
for k, col in zip(range(len(centers)), colors):
    my_members = []
    if k_means_labels.all() == k:
        my_members.append(k_means_labels)
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
ax.set_title('KMeans')
plt.show()
