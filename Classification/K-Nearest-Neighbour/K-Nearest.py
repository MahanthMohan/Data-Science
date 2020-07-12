import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import neighbours

df = pd.read_csv('Data/Dataset/teleCust.csv')
df.head()

df['custcat'].value_counts()
df.hist(column='income', bins = 50)
plt.show()

X = df[['custcat']].values
X[0:5]

y = df[['custcat']].values
y[0:5]

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

X_train, X_test, y_train, y_test = model_selection.train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

k = 4 
neighbours = neighbours.KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

yhat = neighbours.predict(X_test)
yhat[0:5]

print(yhat)
