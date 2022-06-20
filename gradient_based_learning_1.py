import numpy as np
import matplotlib.pyplot as plt


def my_fuc(x):
    y = 3 * (x**2)
    return y


# linspace(start, end, n_point), start부터 end까지를 n_point 개수만큼 나눠서
x = np.linspace(-5, 5, 300)
y = my_fuc(x)

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(x,y)


n_iter = 10
x = 4
a = 0.2     # learning rate
ax.scatter(x, my_fuc(x), color='purple', s=100)

for iter_idx in range(n_iter):
    # backward
    diff = 6*x
    x = x - diff*a
    y = my_fuc(x)

    print('iter idx = ', x)
    ax.scatter(x, my_fuc(x), color='purple', s=100)


ax.tick_params(labelsize = 20)
plt.show()