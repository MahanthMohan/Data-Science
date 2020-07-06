import matplotlib.pyplot as plt # A data visualization library
import pandas as pd # A data manipulation library
import pylab as pl  
import numpy as np # A python math library
from sklearn import linear_model # Linear Model package from sklearn

df = pd.read_csv('Dataset/FuelConsumption.csv')
df.head() # head into the dataset
df.describe() # Summarize the data

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] # Only the datapoints we need for the linear regression
cdf.head(9)

msk = np.random.rand(len(df)) < 0.75
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')

regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit(train_x, train_y)
print('Coefficients: ', regression.coef_)
print('Intercept: ', regression.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regression.coef_[0][0] * train_x + regression.intercept_[0], '-r')
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.show()