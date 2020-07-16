import numpy as np
import matplotlib.pyplot as plt

def plotFunction(f):
    x = np.linspace(-5, 5, 1)
    plt.plot(x,f(x))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

f = lambda x: np.power(x,3) + np.power(x,2) + x + 1
plotFunction(f)
