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


condition = pd.notnull(df[['dateRep', 'cases', 'deaths', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000']])
cdf = df[condition]
cdf.head()

x_data, y_data = (float(cdf['dateRep'].values), float(cdf['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'].values))

fig = plt.figure()
plt.plot(x_data, y_data, 'ro')
plt.xlabel('Time (in days)')
plt.ylabel('COVID cases per 100000')
plt.xticks(None)
plt.yticks(None)
plt.show()
