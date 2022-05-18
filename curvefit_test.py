

from re import X
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
# m = model.sfz_dynamic_model(7706)
tester = arange(.1, 1, .1)
tester = [.3]
for s in tester:
    mse = 0
    ctr = 0
    for k in range(5):

        fd = arange(0, 2)
        itl = s
        x = arange(0, 2, itl)
        size = 50
        size = int(size/len(x))
        y = [np.mean([np.random.binomial(1, model.get_rho(0, r))
                      for _ in range(size)]) for r in x]
        d = [model.get_rho(0, r) for r in fd]

        cr = arange(model.customer[0].c0, model.customer[0].c1)
        cd = [model.get_rho(0, r) for r in cr]
        try:
            coeffient, _ = curve_fit(objective, x, y)
            c0, c1 = coeffient
            cf = objective(cr, c0, c1)
            mse += sum([abs(cf[i]-cd[i]) for i in range(len(cr))])
            ctr += 1
        except:
            pass
    print(s, mse/ctr, ctr)
    # plt.plot(fd, d, color='red')
    # plt.plot(x, y, color='y')
    # plt.plot(x, objective(x, c0, c1), color='b')
    # plt.show()

    plt.plot(cr, cd, color='red')
    plt.plot(cr, cf, '-o', color='b')
    plt.show()
