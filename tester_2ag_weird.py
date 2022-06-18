
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

plt.style.use('fast')


def single_company_data():
    df = pd.read_excel("data/double_company.xlsx", sheet_name=None)
    xest, yest = [], []
    for sheetname in df.keys():
        with scoping():
            df = pd.read_excel("data/double_company.xlsx",
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


def test(xest, yest):
    selects = []
    selects_nav = []
    mest = []
    mest_nav = []

    for i in range(len(xest)):
        a, b, c, d = Cal(xest[i], yest[i])
        selects.append(a)
        mest.append(b)
        selects_nav.append(c)
        mest_nav.append(d)

    # print(model.th_expected_rev(1))
    # print(selects)
    erev = model.get_expected_rev(selects)
    erev_n = model.get_expected_rev(selects_nav)
    rrev = model.get_estimated_rev(selects, mest)
    rrev_n = model.get_estimated_rev(selects_nav, mest_nav)
    # print(model.get_expected_rev(selects))
    # print(selects_nav)
    # print(model.get_expected_rev(selects_nav))
    return [erev, erev_n, rrev, rrev_n]
    # print(model.get_estimated_rev(selects, mest))
    # print(model.get_estimated_rev(selects_nav, mest_nav))
    return


xests, yests = single_company_data()
res = test(xests, yests)
print(res[0], res[1])


def MeanResult(k):
    res = [test(*custom_gdata(k)) for _ in range(30)]
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


single_company_data()
