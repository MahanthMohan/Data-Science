import pandas as pd # Data manipulation library
from sklearn.preprocessing import StandardScaler # Tranforms the data to have zero mean and unit variance/data normalization 
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D # 3d graphing library
from matplotlib import pyplot as plt
import numpy as np

# Customer data
cust_df = pd.read_csv('Dataset/Cust_Segmentation.csv')
cust_df.head()

# Drop address atribute as it has categorical data (Non-Numerical Data)
df = cust_df.drop('Address', axis=1)
df.head()

# Normalize the dataset
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataset = StandardScaler().fit_transform(X)

# Initialize the algorithm
k_means = KMeans(init='k-means++', n_clusters = 3, n_init=12)
k_means.fit(X)
labels = k_means.labels_
print('The labels for the blobs: \n')
print(labels)

df['K-Means'] = labels
df.head(5)

df.groupby('K-Means').mean()

fig = plt.figure(1)
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=148)

ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:,1], X[:,0], X[:,3], c=labels.astype(np.float))
plt.show()
