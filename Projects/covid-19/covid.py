import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x, Beta_1, Beta_2):
    y = 1/(1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y

df = pd.read_csv('Data/covid.csv')
df.head()
df['dateRep'] = pd.to_datetime(df['dateRep']).dt.strftime("%Y%m%d")

cdf = df[['dateRep', 'cases', 'deaths', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000']]
cdf.head()

x_data, y_data = (cdf['dateRep'].values, cdf['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'])
plt.plot(x_data, y_data, 'ro')
plt.xlabel('Time (in days)')
plt.ylabel('COVID cases per 10000')
plt.show()



