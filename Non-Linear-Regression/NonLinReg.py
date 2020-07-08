import numpy as np
import matplotlib.pyplot as plt


x = np.arange(-5.0, 5.0, 0.1)
k = 6
y = 1/(1 + np.power(2.71828, -k * x))
y_noise = np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x, y, 'indigo') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()