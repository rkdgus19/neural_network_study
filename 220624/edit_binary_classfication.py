import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

def make_ds(N, code):
    # parameter setting
    std = 0.1
    mean_x_list = [0, 0, 1, 1]
    mean_y_list = [0, 1, 0, 1]
    X_list, Y_list = [], []

    for idx in range(4):
        # generate X dataset
        mean_x = mean_x_list[idx]
        mean_y = mean_y_list[idx]

        X_tmp = normal(loc=[mean_x, mean_y],
                       scale=[std, std],
                       size=(N, 2))
        X_list.append(X_tmp)

        # generate Y dataset
        code_idx = code[idx]
        Y_tmp = code_idx * np.ones(shape=(N,))
        Y_list.append(Y_tmp)

    # merge X, Y dataset
    X = np.vstack(X_list)
    Y = np.concatenate(Y_list)

    # shuffle X, Y dataset
    shuffling_idx = np.arange(4*N)
    np.random.shuffle(shuffling_idx)

    X = X[shuffling_idx]
    Y = Y[shuffling_idx]

    return X, Y

def vis_ds(X, Y):
    X_pos, Y_pos = X[Y == 1], Y[Y == 1]
    X_neg, Y_neg = X[Y == 0], Y[Y == 0]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(X_pos[:, 0], X_pos[:, 1],
               color='blue')
    ax.scatter(X_neg[:, 0], X_neg[:, 1],
               color='red')
    ax.tick_params(labelsize=20)
    ax.axvline(x=0, color='gray', linewidth=1)
    ax.axvline(x=0.5, color='gray', linewidth=1,
               linestyle=':')
    ax.axvline(x=1, color='gray', linewidth=1)
    ax.axhline(y=0, color='gray', linewidth=1)
    ax.axhline(y=0.5, color='gray', linewidth=1,
               linestyle=':')
    ax.axhline(y=1, color='gray', linewidth=1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    return ax

##### TEST CODE #####
N = 1000
X, Y = make_ds(N, [0, 0, 1, 0])
ax = vis_ds(X, Y)

w1, w2, b = np.random.normal(0, 1, 3)
n_iter = 4*N
lr = 0.03
w1_list, w2_list, b_list = [w1], [w2], [b]

for iter_idx in range(n_iter):
    # get data sample
    (x1, x2), y = X[iter_idx], Y[iter_idx]

    # forward propagation
    z = w1*x1 + w2*x2 + b
    pred = 1/(1 + np.exp(-z))
    J = -1*(y*np.log(pred) +
            (1-y)*np.log(1-pred))

    # differential coefficients
    dz_dw1, dz_dw2, dz_db = x1, x2, 1
    dpred_dz = pred*(1-pred)
    dJ_dpred = (pred-y)/(pred*(1-pred))

    # backpropagation
    dJ_dz = dJ_dpred * dpred_dz
    dJ_dw1 = dJ_dz * dz_dw1
    dJ_dw2 = dJ_dz * dz_dw2
    dJ_db = dJ_dz * dz_db

    # parameter update
    w1 = w1 - lr*dJ_dw1
    w2 = w2 - lr*dJ_dw2
    b = b - lr*dJ_db

    w1_list.append(w1)
    w2_list.append(w2)
    b_list.append(b)


x1_lim = ax.get_xlim()
x2_lim = ax.get_ylim()
print(x1_lim, x2_lim)
x1 = np.linspace(x1_lim[0], x1_lim[1])
x2 = np.linspace(x2_lim[0], x2_lim[1])

X1, X2 = np.meshgrid(x1, x2)
DB = (w1*X1 + w2*X2 + b > 0).astype(int)

ax.contourf(X1, X2, DB, cmap='bwr_r',
            alpha=0.3)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(w1_list, color='r')
ax.plot(w2_list, color='g')
ax.plot(b_list, color='b')

# X1, X2 = np.meshgrid(x1, x2)
# Z = w1*X1 + w2*X2 + b
# A = 1/(1 + np.exp(-Z))
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(projection='3d')
#
# ax.plot_wireframe(X1, X2, A)
plt.show()
