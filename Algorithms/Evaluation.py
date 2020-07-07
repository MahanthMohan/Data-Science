import math
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

def RMSE(test_y, test_y_hat):
		error = 0
		for i in range(len(test_y)):
			error = error + math.pow((test_y[i] - test_y_hat[i]), 2)
		error = math.pow(np.mean(error), 0.5)
		return error

def RAE(test_y, test_y_hat):
	RAE = 0
	for j in range(len(test_y)):
		RSE = RSE + math.pow((test_y[j] - test_y_hat[j]), 2)/(test_y[j] - np.mean(test_y))
	return RSE

def RSE(test_y, test_y_hat):
	RSE = 0
	for j in range(len(test_y)):
		RSE = RSE + math.pow((test_y[j] - test_y_hat[j]), 2)/math.pow((test_y[j] - np.mean(test_y)), 2)
	return RSE

def R2(test_y, test_y_hat):
	RSE = 0
	for j in range(len(test_y)):
		RSE = RSE + math.pow((test_y[j] - test_y_hat[j]), 2)/math.pow((test_y[j] - np.mean(test_y)), 2)
	R2_score = 1 - RSE
	return R2_score