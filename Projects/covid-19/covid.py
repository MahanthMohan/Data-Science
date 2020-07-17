import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def function(x,a,b,c,d):
    return a*(x**3) + b*(x**2) + c*x + d

df = pd.read_csv('Data/covid.csv')
df.head()
df['dateRep'] = pd.to_datetime(df['dateRep']).dt.strftime("%Y%m%d")


condition = pd.notnull(df[['dateRep', 'cases', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000']])
cdf = df[condition]
cdf.head()

x_data, y_data = (cdf['dateRep'].values, cdf['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'].values)

xdata, ydata = ([],[])
for value in x_data:
    res = float(value)/float(max(x_data))
    xdata.append(res)

for value in y_data:
    ret = float(value)/float(max(y_data))
    ydata.append(ret)

coef = curve_fit(function, xdata, ydata)[0]
print('The coefficients are: ')
print(coef)

fig = plt.figure()
x = np.linspace(0, len(x_data), 1)
plt.plot(x_data, y_data, 'ro', label='data')
plt.plot(x, function(x, *coef), linewidth=4.0, label='model')
plt.legend(loc='best')
plt.xlabel('Time (in days)')
plt.ylabel('COVID cases per 100000')
plt.show()
