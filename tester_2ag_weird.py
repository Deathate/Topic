
from rho_simulate_weird import Fitting
import sfztest_1agent_acc as model
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import collections
from scoping import scoping
from scipy.stats import bernoulli
import pandas as pd
from decimal import Decimal
import itertools as it

plt.style.use('fast')


def double_company_data():
    df = pd.read_excel("data/double_company.xlsx", sheet_name=None)
    xest, yest = [], []
    for sheetname in df.keys():
        with scoping():
            df = pd.read_excel("data/double_company.xlsx",
                               sheet_name=sheetname)
            hist = df.iloc[2:, [1, 2, 3]]
            hist.columns = ['an1', 'an2', 'result']
            hist["an1"] = hist["an1"].map(lambda x: Decimal(str(round(x, 2))))
            hist["an2"] = hist["an2"].map(lambda x: Decimal(str(round(x, 2))))
            hist["result"] = hist["result"].map(lambda x: int(x))
            arr = sorted(hist["an1"].drop_duplicates())
            xest.append(list(it.product(arr, arr)))
            yest.append([len(hist[(hist["an1"] == x[0]) & (hist["an2"] == x[1]) & (hist["result"] == 1)]) /
                         len(hist[(hist["an1"] == x[0])
                             & (hist["an2"] == x[1])])
                         for x in xest[-1]])
    xest = list(
        map(lambda x: list(map(lambda x: (float(x[0]), float(x[1])), x)), xest))
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


def Cal(xest, yest):
    x = xest

    f = Fitting(xest, yest)
    best = x[np.argmax([x[i] * f(x[i]) for i in range(len(x))])]

    best_navid = np.argmax([x[i] * yest[i] for i in range(len(x))])
    best_nav = x[best_navid]
    # plt.plot(xest, yest, 'o')
    # plt.plot(xest, f(xest))
    # plt.show()
    return best, f(best),  best_nav, yest[best_navid]


def test(xests, yests):
    selects = []
    selects_nav = []
    mest = []
    mest_nav = []

    for i, _ in enumerate(xests):
        k = []
        s, r = np.array(xests[i]), np.array(yests[i])
        for x in np.unique(s[:, 0]):
            xest = s[s[:, 1] == x]
            yest = r[s[:, 1] == x]
            a, b, c, d = Cal(xest[:, 0], yest)
            k.append((a, b, c, d, a*b, c*d))
        fitr = max(k, key=lambda x: x[4])
        nr = max(k, key=lambda x: x[5])
        selects.append(fitr[0])
        mest.append(fitr[1])
        selects_nav.append(nr[2])
        mest_nav.append(nr[3])

    erev = model.get_expected_rev(selects)
    erev_n = model.get_expected_rev(selects_nav)
    rrev = model.get_estimated_rev(selects, mest)
    rrev_n = model.get_estimated_rev(selects_nav, mest_nav)
    model.th_expected_rev(1)
    print(erev, erev_n)
    print(rrev, rrev_n)
    print(selects)
    print(selects_nav)

    return [erev, erev_n, rrev, rrev_n]


xests, yests = double_company_data()
test(xests, yests)
