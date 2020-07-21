import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs
from matplotlib.pyplot.cm import nipy_spectral as spectral

x, y = make_blobs(n_samples=50, centers=[[4,4], [-2,-1], [1,1], [10,4]], cluster_std=0.9)
plt.scatter(x[:,0], x[:, 1], marker='o')

# 2 arguments - n_clusters and linkage
model = AgglomerativeClustering(4, 'average')
model.fit(x,y)

plt.figure()
x_min, x_max = np.min(x, axis=0), np.max(x,axis=0)
x = (x - x_min)/(x_max - x_min)

for i in range(len(x)):
    plt.text(x[i,0], x[i,1], str(y[i]), 
    color=spectral(model.labels_[i]/10)), 
    fontdict={'weight': 'bold', 'size': 9}

plt.xticks(())
plt.yticks(())

# Original Data before clustering
plt.scatter(x[:,0], x[:,1], marker='.')

# Hierarchical Clustering af the data
distance_matrix = distance_matrix(x, x)
z = hierarchy.linkage(dist_matrix, 'complete')
dendro = hierarchy.dendogram(z)
plt.show()

