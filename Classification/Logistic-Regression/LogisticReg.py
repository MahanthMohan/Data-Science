import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as optimize
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_similarity_index

# Use pandas to read the Churn data csv file
churn_df = pd.read_csv('ChurnData.csv')
churn_df.head()

# Grab the necessary characteristics
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
for element in churn_df['churn']:
    element = int(element)
churn_df.head()

# All the X data (Independent Variables)
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]

# All the y data (Dependent variables)
y = np.asarray(churn_df['churn'])
y[0:5]

# Transform the data to have a zero variance and mean
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

# Initialize the train test splitted data 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 0.4)
LogReg = LogisticRegression(C=0.01, solver = 'liblinear').fit(X_train, Y_train)

yhat = LogReg.predict()
yhat_prob = LogReg.predict_proba(X_test)
accuracy_score = jaccard_similarity_score(y_test, yhat)
print(f'The prediction for churn is: {yhat} \n', f'The actual value is: {y} \n', f'The accuracy score for this model is {accuracy_score}')
