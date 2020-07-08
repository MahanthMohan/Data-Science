import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

def sigmoid(x,  β1, β2):
    y = 1/(1 + np.exp(-β1 * (x - β2)))
    return y

df = pd.read_csv('Data/Dataset/china_gdp.csv')
df.head(10) # Look into the dataset's first 10 rows

x_data, y_data = (df['Year'].values, df['Value'].values) # Get the values of the two columns (Year, Value)
xdata, ydata = (x_data/max(x_data), y_data/max(y_data)) # normalize the dataset
popt = optimize.curve_fit(sigmoid, xdata, ydata) # Use the scipy algorithm to optimize the sigmoid curve
x = np.linspace(1960, 2015, 55)
x = x/max(x) 
plt.plot(xdata, ydata, 'ro')
y = sigmoid(x, popt[0][0], popt[0][1])
plt.plot(x, y, 'indigo')
plt.title('China GDP')
plt.xlabel('Year')
plt.ylabel('GDP (in tens of trillion USD)')
plt.show()