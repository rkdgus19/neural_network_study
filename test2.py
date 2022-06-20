import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

N = 200

# mean = 0, standard deviation = 1
# N points
X = np.random.normal(loc=0, scale=1, size=(N,))

# target weight, target bios
tw, tb = 2, 1
Y = tw * X + tb
noise = 0.3 * np.random.normal(0, 1, (N,))
Y = Y + noise  # noise injection

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X, Y, alpha=0.5, s=50)

## Initial model setting
w, b = -2, -1
x_lim = np.array(ax.get_xlim())

y_lim = w*x_lim+b
ax.plot(x_lim, y_lim, color='red')


## Training
n_iter = 20
learning_rate = 0.1
cmap = cm.get_cmap('rainbow', lut=n_iter)
w_list, b_list = [w], [b]


for iter_idx in range(n_iter):
    x, y = X[iter_idx], Y[iter_idx]

    pred = w*x + b              # predict output
    diff_w = -2*x*(y-pred)      # backward w
    diff_b = -2*(y-pred)        # backward b

    w = w - learning_rate*diff_w        # w 갱신
    b = b - learning_rate*diff_b        # b 갱신

    y_lim = w * x_lim + b
    ax.plot(x_lim, y_lim, color=cmap(iter_idx), alpha=0.5)

# dataset visualization

plt.show()

fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes
