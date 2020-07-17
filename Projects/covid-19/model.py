import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def function(x,a,b,c,d):
    return a*(x**3) + b*(x**2) + c*x + d

df = pd.read_csv('Data/Dataset/covid.csv')
df.head()

condition = pd.notnull(df[['Day', 'cases', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000']])
cdf = df[condition]
cdf.head()

# Initialize the x and y values
x_data, y_data = (cdf['Day'].values, cdf['cases'].values)
xdata, ydata, leng = (x_data/max(x_data), y_data/max(y_data), len(x_data))

# Model (Predicting cases using the trend shown by the data)
popt, pcov = curve_fit(function, xdata, ydata)
# print('The coefficients for the cubic equation are: ')
# print(popt)
x = np.linspace(0, leng, leng)
x = x/max(x) # Data Normalization
plt.plot(xdata, ydata, 'ro')
plt.plot(x, function(x, *popt), 'indigo')
plt.title('The Covid-19 model for USA')
plt.xlabel('Time (in days)')
plt.ylabel(f'Number of Cases in {leng} days')
plt.xticks(x, ' ')
plt.legend(('cases', 'curve fit (normalized values (0 --> 1))'), loc='best')
plt.figtext(0.15, 0.72, 'Scale: (0 --> 823), (1 --> 67717)', fontsize=10)
plt.show()
