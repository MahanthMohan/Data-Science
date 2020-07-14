import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as optimize
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score

# Use pandas to read the Churn data csv file
churn_df = pd.read_csv('Dataset/ChurnData.csv')
churn_df.head()

# Grab the necessary characteristics
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
for element in churn_df['churn']:
    element = int(element)
churn_df.head()

# All the X data (Independent Variables)
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])

# All the y data (Dependent variables)
y = np.asarray(churn_df['churn'])

# Transform the data to have a zero variance and mean
X = preprocessing.StandardScaler().fit(X).transform(X)

# Initialize the train test splitted data 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2, random_state = 4)
LogReg = LogisticRegression(C=0.01, solver = 'liblinear').fit(X_train, y_train)

yhat = LogReg.predict(X_test)
yhat_prob = LogReg.predict_proba(X_test)
accuracy_score = jaccard_score(y_test, yhat)

print(f'The probabilties for Class 0 and Class 1: \n{yhat_prob} \n', f'The accuracy score for this model is {accuracy_score}')
