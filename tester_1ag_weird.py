
from rho_simulate_weird import Fitting, linear, quadratic
import sfztest_1agent_acc as model
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import collections
from scoping import scoping
from scipy.stats import bernoulli
import pandas as pd
from decimal import Decimal

plt.style.use('fast')


def single_company_data():
    df = pd.read_excel("data/single_company.xlsx", sheet_name=None)
    xest, yest = [], []
    for sheetname in df.keys():
        with scoping():
            df = pd.read_excel("data/single_company.xlsx",
                               sheet_name=sheetname)
            hist = df.iloc[2:, [1, 2]]
            hist.rename(columns={'c0': 'an',
                                 'c1': 'result'}, inplace=True)
            hist["an"] = hist["an"].map(lambda x: Decimal(str(round(x, 2))))
            hist["result"] = hist["result"].map(lambda x: int(x))

            xest += [sorted(hist["an"].drop_duplicates())]
            yest += [[len(hist[(hist["an"] == x) & (hist["result"] == 1)]) / len(hist[(hist["an"] == x)])
                      for x in xest[-1]]]
    xest = list(map(lambda x: list(map(lambda x: float(x), x)), xest))
    return xest, yest


def custom_gdata(k):

    def create(i):
        x = model.arange(model.customer[i].c0, model.customer[i].c1, 0.05)
        data = collections.defaultdict(list)
        with scoping():
            for _ in range(len(x)*k):  # len(x)*10
                an = np.random.choice(x)
                data[an].append(bernoulli.rvs(model.get_rho(i, an)))
        est = {x: sum(data[x])/len(data[x]) for x in data}
        est = dict(sorted(est.items()))
        xest = np.array(list(est.keys()))
        yest = np.array(list(est.values()))
        return xest, yest
    xests = []
    yests = []
    for i in range(10):
        a, b = create(i)
        xests.append(a)
        yests.append(b)
    return xests, yests


def Cal(xest, yest, robust=True, ax=None):
    f = Fitting(xest, yest, ax, robust, [linear, quadratic])
    best = xest[np.argmax([xest[i] * f(xest[i]) for i in range(len(xest))])]

    f_nav = Fitting(xest, yest, None, True, [])
    best_nav = xest[np.argmax([xest[i] * f_nav(xest[i])
                              for i in range(len(xest))])]

    return best, f(best),  best_nav, f_nav(best_nav)


def test(xest, yest, robust, graph=False):
    selects = []
    selects_nav = []
    mest = []
    mest_nav = []
    if graph:
        fig, ax = plt.subplots(2, 5)
        fig.set_size_inches(1920/100, 800/100)
        fig.tight_layout()
    for i in range(len(xest)):
        axu = None
        if graph:
            axu = ax[int(i/5)][i % 5]
            x_left, x_right = axu.get_xlim()
            y_low, y_high = axu.get_ylim()
            axu.set_aspect(abs((x_right-x_left)/(y_low-y_high)*.8))
            axu.plot(xest[i], [model.get_rho(i, x)
                     for x in xest[i]], color="r")
        a, b, c, d = Cal(xest[i], yest[i], robust, axu)
        selects.append(a)
        mest.append(b)
        selects_nav.append(c)
        mest_nav.append(d)
    if graph:
        plt.show()
    erev = model.get_expected_rev(selects)
    erev_n = model.get_expected_rev(selects_nav)
    rrev = model.get_estimated_rev(selects, mest)
    rrev_n = model.get_estimated_rev(selects_nav, mest_nav)

    return [erev, erev_n, rrev, rrev_n]


def MeanResult(k):
    res = [test(*custom_gdata(k), robust=k > 100) for _ in range(30)]
    a, b, c, d = np.mean([x[0] for x in res]), np.mean([x[1] for x in res]), np.mean(
        [x[2] for x in res]), np.mean([x[3] for x in res])
    return [a, b, c, d]


def RangeTest():
    sample = [10, 20, 30, 40, 50, 60, 100, 200, 500]
    s = [MeanResult(x) for x in sample]
    _, ax = plt.subplots()
    ax.plot(sample, [x[0] for x in s], '-o')
    ax.plot(sample, [x[1] for x in s], '-o')
    ax.set_xticks(sample)
    plt.show()


# RangeTest()

xests, yests = single_company_data()
res = test(xests, yests, robust=True, graph=False)
print(res[0], res[1])

# xests, yests = custom_gdata(40)
# res = test(xests, yests, robust=False, graph=True)
# print(res[0], res[1])
