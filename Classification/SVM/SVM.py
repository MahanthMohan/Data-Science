import pandas as pd
import pylab as pl
import numpy as np
from scipy import optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import svm

cell_df = pd.read_csv('Data/cell_samples.csv')
cell_df.head()

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
for element in cell_df['BareNuc']:
    element = int(element)

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]

cell_df['Class'] = cell_df['Class']
for element in cell_df['Class']:
    element = int(element)
y = np.asarray(cell_df['Class'])
y[0:5]

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state = 4)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
yhat[0:5]

print('The Classified Results using SVM are: ')
print(yhat)
