import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def sigmoid(x, Beta_1, Beta_2):
    f = lambda x, Beta_1, Beta_2: 1/(1 + np.exp(-Beta_1 * (x - Beta_2)))
    y = f(x, Beta_1, Beta_2)
    return y

df = pd.read_csv('Data/covid.csv')
df.head()
df['dateRep'] = pd.to_datetime(df['dateRep']).dt.strftime("%Y%m%d")

cdf = df[['dateRep', 'cases', 'deaths', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000']]
cdf.head()

x_data, y_data = (cdf['dateRep'].values, cdf['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'])

# Data Normalization
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

# Optimization of the weight vector
popt = curve_fit(sigmoid, xdata, ydata)
print(popt)

fig = plt.figure(1)
ax = fig.add_subplot()
ax.plot(x_data, y_data, 'ro')
ax.xlabel('Time (in days)')
ax.ylabel('COVID cases per 100000')
ax.set_xticks(())
ax.set_yticks(())
plt.show()

