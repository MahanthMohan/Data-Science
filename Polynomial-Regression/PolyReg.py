import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

df = pd.read_csv('Data/Dataset/FuelConsumption.csv')
df.head()

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(9)

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, c='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()

msk = np.random.rand(len(df)) < 0.75
train = cdf[msk]
test = cdf[~msk]

train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)

clf = linear_model.LinearRegression()
train_y = clf.fit(train_x_poly, train_y)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, c='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0] + clf.coef_[0][1] * XX + clf.coef_[0][2] * (XX**2)
plt.plot(XX, yy, '-r')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')
plt.show()