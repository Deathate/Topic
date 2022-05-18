

import sfztest_1agent_acc as model
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def arange(c0, c1, step=0.01):
    return list(map(lambda x: round(x, 2), np.linspace(
        c0, c1, int((c1 - c0)/step)+1, endpoint=True)))


def objective(x, a, b):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    return sigmoid(np.poly1d([a, b])(x))


model.N2 = 5
m = model.sfz_dynamic_model(7706)


x = m.LoadHistdata()[0]
x = dict(sorted(x.items()))

for r in x:
    print(
        f"{r:<5} {len(x[r]):<5} est: {round(np.mean(x[r]),4):<5}  real: {round(model.get_rho(0, r),4)}")
    print("-----------------")
y = [np.mean(x[r]) for r in x]

xspace = [r for r in x]
coeffient, _ = curve_fit(objective, xspace, y)
c0, c1 = coeffient

# d = [model.get_rho(0, r) for r in xspace]
# plt.plot(xspace, d, '--', color='red')
# plt.plot(xspace, y, '--', color='y')
# plt.plot(xspace, objective(xspace, c0, c1), '--', color='b')
# plt.show()
