import matplotlib.pyplot as plt # A data visualization library
import pandas as pd # A data manipulation library  
import numpy as np # A python math library

df = pd.read_csv('Dataset/FuelConsumption.csv')
df.head() # head into the dataset
df.describe() # Summarize the data

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']] # Only the datapoints we need for the linear regression
cdf.head(9)

plt.title('CO2 Emissions')

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, c="blue")
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions')

plt.show()
