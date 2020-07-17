import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_csv('Data/dataset.csv')
df.head()
df['dateRep'] = pd.to_datetime(df['dateRep']).dt.strftime("%Y%m%d")


condition = pd.notnull(df[['dateRep', 'cases', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000']])
cdf = df[condition]
cdf.head()

x_data, y_data = (cdf['dateRep'].values, cdf['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'].values)

fig = plt.figure()
plt.plot(x_data, y_data, 'ro')
plt.xlabel('Time (in days)')
plt.ylabel('COVID cases per 100000')
plt.show()

