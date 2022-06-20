import numpy as np


class Phone:
    def __init__(self, w=None):
        self.w = w

        self._init_params()

    def _init_params(self):
        if self.w is None:
            self.w = np.random.randint(0, 2, 4)

    def forward(self, x):
        if not self.is_v_available(x):
            z = -1
        else:
            z = np.dot(self.w, x)
        return z

    def is_v_available(self, v):
        if len(v) != 4:
            return False
        else:
            return True


phone1 = Phone(w=[1, 1, 0, 0])
phone2 = Phone(w=[0, 1, 0, 1])
phone3 = Phone(w=[0, 0, 1, 1])
phones = [phone1, phone2, phone3]

x = np.array([1, 1, 0, 0, 1])
for i, temp_phone in enumerate(phones):
    r = temp_phone.forward(x)
    print('[' + str(i) + '] :', temp_phone.w, ' / ', r)
