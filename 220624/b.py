import numpy as np

a = np.linspace(0, 6, 7)
b = np.linspace(6, 0, 7)

X, Y = np.meshgrid(a, b)
print(X)
print(Y)