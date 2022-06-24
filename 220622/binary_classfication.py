import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class DatasetGenerator:
    def __init__(self, N=200, w=None, b=None):
        self.N = N
        self.w, self.b = w, b

        self._init_params()
        self._make_dataset()

    def _init_params(self):
        if self.w == None: self.w = np.random.randint(-5,5,1)
        if self.b == None: self.b = np.random.randint(-5,5,1)

    def _make_dataset(self):
        self.X = np.random.normal(0, 1, (self.N, ))
        y_hat = (self.w* self.X + self.b)
        y = 1/(1+np.exp(y_hat))
        self.Y = np.where(y < 0.5, 0, 1)
        print(self.Y)

    def get_ds(self): return self.X, self.Y

    def visualize_ds(self):
        flg, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.X, self.Y, alpha=0.4, s=100)
        return ax


class Sigmoid:
    def __init__(self): pass

    def forward(self, x):
        self.x = x
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dJ_):
        dout = dJ_ * self.out * (1 - self.out)
        return dout


class Affine:
    def __init__(self, w=None, b=None):
        self.dz_db = None
        self.dz_dw = None
        self.x = None
        self.w, self.b = w, b
        self.lr = 0.1       # learning rate

        self._init_params()

    def _init_params(self):
        if self.w is None:
            self.w = np.random.uniform(-3, 3, 1)
        if self.b is None:
            self.b = np.random.uniform(-3, 3, 1)

    def forward(self, x):
        self.x = x
        z = self.w * self.x + self.b
        self.dz_dw = x
        self.dz_db = 1
        return z

    def backward(self, dJ_dz):
        dJ_dw = dJ_dz * self.dz_dw
        dJ_db = dJ_dz * self.dz_db

        self.w = self.w - self.lr * dJ_dw
        self.b = self.b - self.lr * dJ_db

    def get_params(self): return self.w, self.b


class Model:
    def __init__(self):
        # import the Affine, Activation Class
        #
        self.affClass = Affine()
        self.actClass = Sigmoid()

    def forward(self, x):
        self.x = x
        x = self.actClass.forward(self.affClass.forward(x))
        return x

    def backward(self, dJ_dpred):
        dj_dz = self.actClass.backward(dJ_dpred)
        self.affClass.backward(dj_dz)

    def get_params(self):
        return self.affClass.get_params()




class BCE:
    # pred = ax + b = y_hat
    def __init__(self): pass

    def forward(self, y, pred):
        self.y, self.pred = y, pred
        #  ylog(pred)-(1-y)log(1-pred)
        J = -((y*np.log(pred)) + ((1-y)*np.log(1-pred)))
        return J

    def backward(self):
        # J의 미분값 =(pred-y)/(pred*(1-pred))
        dJ_pred = (self.pred-self.y)/(self.pred*(1-self.pred))
        return dJ_pred


if __name__ == "__main__":
    ds_gen = DatasetGenerator(N=1000, w=3, b=-2)
    X, Y = ds_gen.get_ds()
    ax_ds = ds_gen.visualize_ds()

    x_lim = np.array(ax_ds.get_xlim())

    model = Model()
    loss_function = BCE()

    # 학습 가능하도록 구현하기(다시 보기)
    n_iter = 1000
    lr = 1e-2
    losses = []

    db_list = []
    w_list, b_list = [model.affClass.w], [model.affClass.b]

    cmap = cm.get_cmap('rainbow', lut=n_iter)
    for iter_idx in range(n_iter):
        x, y = X[iter_idx], Y[iter_idx]

        pred = model.forward(x)
        J = loss_function.forward(y, pred)
        losses.append(J)

        dJ_dpred = loss_function.backward()
        model.backward(dJ_dpred)

        w, b = model.get_params()

        w_list.append(w)
        b_list.append(b)
        db_list.append(-b / w)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(losses)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(db_list)
    ax.plot(w_list, label='weight', color='red')
    ax.plot(b_list, label='bias', color='blue')
    ax.axhline(y=ds_gen.w, color='red', linestyle=':')
    ax.axhline(y=ds_gen.b, color='red', linestyle=':')

    ax.legend()
    plt.show()