import numpy as np
import matplotlib.pyplot as plt


class LR_DsGenerator:
    def __init__(self, N=200, w=None, b=None):
        self.N = N
        self.w, self.b = w, b

        self._init_params()
        self._make_dataset()

    """
    파이썬에서는 private, protected...를 명시적(Explicit)으로 표현한다.
    
    __methodname: private method
    _methodname: protected method
    methodname: public method
    """

    def _init_params(self):     # is not user method
        # if w|b is none then w|b is random int
        if self.w is None:
            self.w = np.random.randint(-5, 5, 1)
        if self.b is None:
            self.b = np.random.randint(-5, 5, 1)

    def _make_dataset(self):    # is not user method
        self.X = np.random.normal(0, 1, (self.N, ))
        self.Y = self.w * self.X + self.b
        noise = 0.3*np.random.normal(0, 1, (self.N, ))
        self.Y = self.Y + noise

    def get_ds(self): return self.X, self.Y

    def visualize_ds(self):
        _, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(self.X, self.Y, alpha=0.4, s=100)
        return ax


class Model:
    def __init__(self, w=None, b=None):
        self.x = None
        self.w, self.b = w, b

        self._init_params()

    def _init_params(self):
        if self.w is None:
            self.w = np.random.uniform(-3, 3, 1)
        if self.b is None:
            self.b = np.random.uniform(-3, 3, 1)

    def forward(self, x):
        self.x = x          # backward의 dpred_dw 값이 x이므로, 여기서 x값을 저장한다.
        pred = self.w * self.x + self.b
        return pred

    def backward(self, dJ_dpred, lr):
        dpred_dw = self.x
        dpred_db = 1

        dJ_dw = dJ_dpred * dpred_dw
        dJ_db = dJ_dpred * dpred_db

        self.w = self.w - lr*dJ_dw
        self.b = self.b - lr*dJ_db

    def get_params(self): return self.w, self.b


# loss function: Squared Error(SE)
class SquaredError:
    def __init__(self): pass

    def forward(self, y, pred):
        self.y, self.pred = y, pred
        J = (y-pred)**2
        return J

    def backward(self):
        dJ_dpred = -2*(self.y - self.pred)
        return dJ_dpred


ds_gen = LR_DsGenerator(N=200, w=3, b=-2)
X, Y = ds_gen.get_ds()
ax_ds = ds_gen.visualize_ds()

model = Model()
loss_func = SquaredError()

x_lim = np.array(ax_ds.get_xlim())
w, b = model.get_params()
y_lim = w*x_lim + b


n_iter = 200
lr = 0.01
losses = []
w_list, b_list = [model.w], [model.b]

for iter_idx in range(n_iter):
    x, y = X[iter_idx], Y[iter_idx]

    # forward
    pred = model.forward(x)
    J = loss_func.forward(y, pred)
    losses.append(J)

    # backward
    dJ_dpred = loss_func.backward()
    model.backward(dJ_dpred, lr)

    w_list.append(model.w)
    b_list.append(model.b)

# visualization
# _, ax = plt.subplots(figsize=(10, 5))
# ax.plot(losses)
#
# _, ax = plt.subplots(1, 1, figsize=(10,5))
# ax.plot(w_list, label='weight', color='red')
# ax.plot(b_list, label='bias', color='blue')
# ax.axhline(y=ds_gen.w, color='red', linestyle=':')
# ax.axhline(y=ds_gen.b, color='blue', linestyle=':')
#
# ax.legend()
plt.show()