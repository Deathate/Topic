

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings
import matplotlib.pyplot as plt
from collections import namedtuple
warnings.simplefilter("ignore")

cusseed = list(range(1000))
cid = [0] * 300 + [1] * 400 + [2] * 300

SAVEFIG = True


class Setting:
    pass


setting = None


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
    [0, 3]), np.array([.5, -.5]), method="dogbox")
# plt.plot(arange(0, 6, .01), ii_func(arange(0, 6, .01), *iip))
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
        T = 0.3
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

        if select == 1:
            ctr1 += 1
            amount[group] += 1
        elif select == 2:
            ctr2 += 1
            amount[group] -= 1
    if change:
        for i, x in enumerate(amount):
            setting.cstates[i] -= x/400
            setting.cstates[i] -= (a0-a1)/2
            setting.cstates[i] = np.clip(setting.cstates[i], 0, 1)
    S = namedtuple("h", "a0,a1,copy0,copy1,rev0,rev1,rate")
    sd = [a0, a1, ctr1, ctr2, int(-(300000 + ctr1 * 1500) + ctr1 * 1000 * (setting.II + ii_func(a2, *iip) - a0)),
          int(-(300000 + ctr2 * 1500) + ctr2 * 1000 * (setting.II + ii_func(a2, *iip) - a1)), a2]
    return S._make(sd)


def Starter():
    global setting
    setting = Setting()
    setting.c1place = 0
    setting.c2place = 1
    setting.II = 6
    setting.cstates = [.5] * 3
    hist = []
    return hist


def Result(hist, name, show):
    hist = np.array(hist)
    if SAVEFIG or show:
        plt.style.use('ggplot')
        fig, ax = plt.subplots(4, 1)
        ax[0].plot(range(1000), hist[:, 4], label="R1", color="orange")
        ax[0].plot(range(1000), hist[:, 5],
                   label="R2", color="blue")
        ax[1].axhline(np.mean(hist[:, 4]), color="orange", lw=3)
        ax[1].axhline(np.mean(hist[:, 5]), color="blue", lw=3)
        ax[2].plot(range(1000), hist[:, 0], color="orange")
        ax[2].plot(range(1000), hist[:, 1], color="blue")
        ax[2].plot(range(1000), rates, color="green",
                   alpha=.3, label="federal rate")
        ax[3].plot(range(1000), hist[:, 2], color="orange")
        ax[3].plot(range(1000), hist[:, 3], color="blue")
        fig.legend()
        fig.set_size_inches(18, 10)
        fig.suptitle(name, size=20)
        fig.savefig("p3/pct/" + name)
    if show:
        plt.show()
    mean1 = np.mean(hist[:, 4])
    mean2 = np.mean(hist[:, 5])
    if mean1 > mean2:
        return 1
    elif mean1 < mean2:
        return 2
    else:
        return 3


def ClearGraph():
    import glob
    import os
    for f in glob.glob("p3/pct/*"):
        os.remove(f)


rates = Rate_Create()
