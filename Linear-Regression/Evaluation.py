import math
import pandas as pd
from sklearn import linear_model
import numpy as np

def MAE(test_y, test_y_hat):
	test_y_sum = 0
	test_y_hat_sum = 0
	for i in range(len(test_y)):
		test_y_sum = test_y_sum + test_y[i]
		test_y_hat_sum = test_y_hat_sum + test_y_hat[i]
	error = np.mean(np.abs((test_y_sum - test_y_hat_sum)))
	return error

def MSE(test_y, test_y_hat):
		error = 0
		for i in range(len(test_y)):
			error = error + math.pow((test_y[i] - test_y_hat[i]), 2)
		error = np.mean(error)
		return error

def R2(test_y, test_y_hat):
	RSE = 0
	for i in range(len(test_y)):
		RSE = RSE + math.pow((test_y[i] - test_y_hat[i]), 2)/math.pow((test_y[i] - np.mean(test_y)), 2)
	R2_score = 1 - RSE
	return R2_score

df = pd.read_csv('Dataset/FuelConsumption.csv')
df.head() # head into the dataset

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] # Only the datapoints we need for the linear regression
cdf.head(9)

msk = np.random.rand(len(df)) < 0.75
train = cdf[msk]
test = cdf[~msk]

regression = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regression.fit(train_x, train_y)

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = regression.predict(test_x)

print('The mean absolute error is ' + str(MAE(test_y, test_y_hat)))
print('The mean square error is ' + str(MSE(test_y, test_y_hat)))
print('The R2 score is ' + str(R2(test_y, test_y_hat)))