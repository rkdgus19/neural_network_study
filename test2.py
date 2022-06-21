import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# dataset preparation
# dataset generation
N = 200
# mean = 0, standard deviation = 1
# N points
X = np.random.normal(loc=0, scale=1, size=(N, ))
t_w, t_b = -2, 3
Y = t_w * X + t_b
noise = 0.3*np.random.normal(0, 1, (N, ))
Y = Y + noise                                           # noise injection

# dataset visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(X, Y, alpha=0.4, s=100)

# Initial model setting
w, b = np.random.randint(-10, 10, (2, ))

x_lim = np.array(ax.get_xlim())
y_lim = w*x_lim + b
ax.plot(x_lim, y_lim, color='purple', linewidth=3)


# Training
n_iter = 200
learning_rate = 0.01
cmap = cm.get_cmap('rainbow', lut=n_iter)
w_list, b_list = [w], [b]
losses = []
for iter_idx in range(n_iter):
    x, y = X[iter_idx], Y[iter_idx]

    pred = w * x + b
    loss = (y - pred)**2
    diff_w = -2 * x * (y - pred)
    diff_b = -2 * (y - pred)

    w = w - learning_rate * diff_w
    b = b - learning_rate * diff_b

    y_lim = w * x_lim + b
    ax.plot(x_lim, y_lim, color=cmap(iter_idx), linewidth=1, alpha=0.5)

    w_list.append(w)
    b_list.append(b)
    losses.append(loss)

ax.tick_params(labelsize=20)

fig, axes = plt.subplots(2, 1, figsize=(10, 5))
axes[0].axhline(w, color='red', linestyle=':')
axes[0].plot(w_list)
axes[1].axhline(b, color='red', linestyle=':')
axes[1].plot(b_list)

plt.show()
