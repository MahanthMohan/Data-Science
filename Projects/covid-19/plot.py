import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Dataset/covid.csv')
df.head()

condition = pd.notnull(df[['Day', 'cases', 'Cumulative_number_for_14_days_of_COVID-19_cases_per_100000']])
cdf = df[condition]
cdf.head()

# Initialize the x and y values
x, y = (cdf['Day'].values, cdf['cases'].values)
x_data, y_data, leng = (cdf['Day'].values, cdf['Cumulative_number_for_14_days_of_COVID-19_cases_per_100000'].values, len(x))

# Main Figure Initialization
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('COVID-19 models (Cubic relationships)')

# Figure 1 (Total Number of Covid cases)
ax1.plot(x, y, 'bo')
ax1.set_xticks(np.arange(leng), minor=True)
ax1.set_xticklabels(' ')
ax1.set_xlabel('Time (in days)')
ax1.set_ylabel(f'COVID-19 cases in USA')

# Figure 2 (Covid cases per 100000)
ax2.plot(x_data, y_data, 'ro')
plt.xticks(np.arange(leng), ' ')
ax2.set_xlabel(f'Time (in days for {leng} days)')
ax2.set_ylabel('COVID cases per 100000')

# Display the plotted figure with the sub plots
plt.show()