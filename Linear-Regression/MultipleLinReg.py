import matplotlib.pyplot as plt # A data visualization library
import pandas as pd # A data manipulation library
import pylab as pl  
import numpy as np # A python math library
from sklearn import linear_model # Linear Model package from sklearn

df = pd.read_csv('Data/Dataset/FuelConsumption.csv')
df.head() # head into the dataset

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] # Only the datapoints we need for the linear regression
cdf.head(9)

msk = np.random.rand(len(df)) < 0.75
train = cdf[msk]
test = cdf[~msk]

regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit(train_x, train_y)

y_hat = regression.predict(test['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB'])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])