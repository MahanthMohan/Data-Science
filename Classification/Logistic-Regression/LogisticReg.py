import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as optimize
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import LogisticRegression

# Use pandas to read the Churn data csv file
churn_df = pd.read_csv('ChurnData.csv')
churn_df.head()

# Grab the necessary characteristics
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
churn_df['churn'] = churn_df['churn'].astype(int) # Convert churn datas to type <int>
churn_df.head()

# All the X data (Independent Variables)
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

# All the y data (Dependent variable)
y = np.asarray(churn_df['churn'])
y[0:5]

# Transform the data to have a zero variance and mean
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

# Initialize the train test splitted data 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0.4)

