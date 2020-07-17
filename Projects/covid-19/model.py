import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def function(x,a,b,c,d):
    return a*(x**3) + b*(x**2) + c*x + d

df = pd.read_csv('Data/Dataset/dataset.csv')
df.head()
df['dateRep'] = pd.to_datetime(df['dateRep']).dt.strftime("%Y%m%d")

condition = pd.notnull(df[['dateRep', 'cases', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000']])
cdf = df[condition]
cdf.head()

x_data, y_data = (cdf['dateRep'].values, cdf['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'].values)

# Coverting random string values in the data to float and normalizing it
xdata, ydata = ([],[])
for i in range(len(x_data)):
    xdata.append(float(x_data[i])/float(max(x_data)))
    ydata.append(float(x_data[i])/float(max(x_data)))

# Fitting a curve to the data using the curve_fit function from the sklearn.optimize package
coef = curve_fit(function, xdata, ydata)[0]
print('The coefficients are: ')
print(coef)

# Plotting the model
leng = len(x_data)
x = np.linspace(-120, 120, 120)
fig = plt.figure()
plt.plot(x, function(x, *coef), linewidth=3.0, label='curve model')
plt.legend(loc='best')
plt.xlabel('Time (in days)')
plt.ylabel('COVID cases per 100000')
plt.show()