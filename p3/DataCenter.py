

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
import matplotlib.pyplot as plt
from collections import namedtuple
warnings.simplefilter("ignore")

cusseed = np.random.default_rng(0).integers(1000, size=1000)
randomer = np.random.default_rng(1)
cid = [0] * 300 + [1] * 400 + [2] * 300

SAVEFIG = True


class Setting:
    pass


setting = Setting()


def arange(c0, c1, step=0.01):
    return list(map(lambda x: round(x, 2), np.linspace(
        c0, c1, int((c1 - c0) / step) + 1, endpoint=True)))


def Cus_Create(h):
    rng = np.random.default_rng(cusseed[h])
    customer = [rng.normal(.25, 0.036) for _ in range(300)] + \
        [rng.normal(.85, 0.053) for _ in range(400)] + \
        [rng.normal(1.25, 0.15) for _ in range(300)]
    return customer


# plt.hist([x for x in Cus_Create()], density=True, bins=50)
# plt.show()
# exit()


def Rate_Create():
    df = pd.read_excel("p3/search.xlsx")
    rates = df.iloc[:, 2]
    rates = np.array(rates)
    rates = np.flip(rates)
    rates = rates[0:1000]
    return rates
    # plt.plot(rates)
    # plt.show()
    # exit()


def ii_func(x, a, b, c, d):
    return d / (1 + np.exp(-(np.poly1d([a, b])(x)))) - c


iip, _ = curve_fit(ii_func, np.array(
    [0, 3]), np.array([-1, 1]), method="dogbox")
# plt.plot(arange(0, 3, .01), ii_func(arange(0, 3, .01), *iip))
# plt.show()
# exit()


def Rev(h, a0, a1, change=True):
    ctr1 = 0
    ctr2 = 0
    amount = [0, 0, 0]
    cus = Cus_Create(h)
    a2 = rates[h]
    for i in range(1000):
        threshold = cus[i]
        group = cid[i]
        d1, d2 = a0 - a2, a1 - a2
        select = 0
        T = 1.1
        if d1 < threshold and d2 < threshold:
            continue
        elif d1 < threshold and d2 >= threshold:
            select = 2
        elif d1 >= threshold and d2 < threshold:
            select = 1
        elif (d1 - T * abs(setting.cstates[group] - setting.c1place) > d2 - T * abs(setting.cstates[group] - setting.c2place)):
            select = 1
        elif (d1 - T * abs(setting.cstates[group] - setting.c1place) < d2 - T * abs(setting.cstates[group] - setting.c2place)):
            select = 2
        elif (d1 - T * abs(setting.cstates[group] - setting.c1place) == d2 - T * abs(setting.cstates[group] - setting.c2place)):
            select = randomer.integers(1, 3)

        if select == 1:
            ctr1 += 1
            amount[group] += 1
        else:
            ctr2 += 1
            amount[group] -= 1
    if change:
        for i, x in enumerate(amount):
            if x > 0:
                setting.cstates[i] -= .1
            if x < 0:
                setting.cstates[i] += .1
    S = namedtuple("h", "a0,a1,copy0,copy1,rev0,rev1")
    sd = [a0, a1, ctr1, ctr2, int(ctr1 * 50000 * (setting.II + ii_func(a2, *iip) - a0) / 100),
          int(ctr2 * 50000 * (setting.II + ii_func(a2, *iip) - a1) / 100)]
    setting.cstates = [.5] * 3
    return S._make(sd)


def Starter():
    hist = []
    hist.append(Rev(0, rates[0] + .2, rates[0] + .2, False))
    hist.append(Rev(1, rates[0] + .5, rates[0] + .5, False))
    hist.append(Rev(2, rates[0] + 1, rates[0] + 1, False))
    hist.append(Rev(3, rates[0] + 1.5, rates[0] + 1.5, False))
    hist.append(Rev(4, rates[0] + 1.8, rates[0] + 1.8, False))
    return hist


def Result(hist, name, show):
    hist = np.array(hist)
    if SAVEFIG or show:
        plt.style.use('ggplot')
        fig, ax = plt.subplots(3, 1)
        ax[0].plot(range(1000), hist[:, 4][5:], label="R1", color="blue", lw=2)
        ax[0].plot(range(1000), hist[:, 5][5:],
                   label="R2", color="orange", lw=2)
        ax[1].axhline(np.mean(hist[:, 4][5:]), color="blue", lw=3)
        ax[1].axhline(np.mean(hist[:, 5][5:]), color="orange", lw=3)
        ax[2].plot(range(1000), hist[:, 0][5:], color="blue")
        ax[2].plot(range(1000), hist[:, 1][5:], color="orange")
        fig.legend()
        # fig.set_size_inches(18, 10)
        fig.suptitle(name, size=20)
        fig.savefig("p3/pct/" + name)
    if show:
        plt.show()
    return np.mean(hist[:, 4][5:]) > np.mean(hist[:, 5][5:])


setting.c1place = 0
setting.c2place = 1
setting.II = 2.5
setting.cstates = [.5] * 3
rates = Rate_Create()
