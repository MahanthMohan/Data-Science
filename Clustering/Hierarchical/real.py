import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import scipy
import pylab
import scipy.cluster.hierarchy as hierarchy

cdf = pd.read_csv('Data/Dataset/cars_clus.csv')
cdf.head()

# Clean up the dataset and remove all the null values
cdf[['sales', 'resale', 'type', 'price', 'engine_s','horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
'mpg', 'lnsales']] = cdf[['sales', 'resale', 'type', 'price', 'engine_s','horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap','mpg', 'lnsales']].apply(pd.to_numeric(), errors='coerce')
pdf = pdf.dropna()
cdf = cdf.reset_index(drop=True)
cdf.head()

# Selcting values for the featureset and scaling using min max transform to transform the data to have zero mean and unit variance
featureset = cdf[['engine_s', 'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
x = featureset.values # A numpy array
min_max_scaler = MinMaxScaler()
feature_matrix = min_max_scaler.fit_transform(x)
feature_matrix[:5]

leng = feature_matrix.shape[0]
Z = scipy.zeros([leng, leng])
for i in range(leng):
    for j in range(leng):
        Z[i,j] = scipy.spatial.distance.euclidean(feature_matrix[i], feature_matrix[j])

Linkage = hierarchy.linkage(Z, 'complete')
fig = pylab.figure()
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])))

dendro = hierarchy.dendrogram(X, leaf_label_func=llf, leaf_rotation=0, leaf_font_size=12, orientation='right')
plt.show()
