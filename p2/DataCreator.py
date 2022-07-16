

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import namedtuple
import pickle
import warnings
import os
warnings.simplefilter("ignore")

c1place = 0
c2place = 1
II = 4
rates = []
hist = []
customer_old = {}
STP = 500


def Cus_Create():
    customer = []
    rng = np.random.default_rng(0)
    cstate = namedtuple('r', 'b0dis, b1dis, threshold')
    for i in range(1000):
        r = rng.random()
        if i < 100:
            threshold = rng.normal(.1, 0.016)
        elif i < 400:
            threshold = rng.normal(.15, 0.033)
        else:
            threshold = rng.normal(.35, 0.1)
        customer.append(cstate._make(
            (abs(r - c1place), abs(r - c2place), threshold)))
    return customer
    # pd.DataFrame(customer, columns=['B0dis', 'B1dis', 'Threshold']).to_excel(
    #     "p2/cusdata.xlsx")


def Rate_Create():
    global rates
    df = pd.read_excel("p2/search.xlsx")
    rates = df.iloc[:, 2]
    rates = np.array(rates)
    rates = np.flip(rates)
    rates = rates[0:1000]
    # plt.plot(rates)
    # plt.show()


def ii_func(x, a, b, c, d):
    return d / (1 + np.exp(-(np.poly1d([a, b])(x)))) - c


iip, _ = curve_fit(ii_func, np.array(
    [0, 3]), np.array([.5, -.5]), method="dogbox")


def Rev(a0, a1, a2, customer, alter=True):
    ctr = 0
    for j in range(1000):
        c = customer[j]
        d1, d2 = a0 - a2, a1 - a2
        select = 0
        T = 1.1
        if d1 < c.threshold and d2 < c.threshold:
            continue
        elif d1 < c.threshold and d2 >= c.threshold:
            select = 2
        elif d1 >= c.threshold and d2 < c.threshold:
            select = 1
        elif (d1 - T * c.b0dis >= d2 - T * c.b1dis):
            select = 1
        else:
            select = 2
        if select == 1:
            ctr += 1
        if alter:
            if select == 1:
                customer[j] = c._replace(b0dis=abs(c.b0dis - .1),
                                         b1dis=abs(c.b1dis + .1))
            elif select == 2:
                customer[j] = c._replace(b0dis=abs(c.b0dis + .1),
                                         b1dis=abs(c.b1dis - .1))
            else:
                print("error")

    return ctr, int(ctr * 100000 * (II + ii_func(a2, *iip) - a0) / 100)


def Hist_Create():
    global hist
    rng = np.random.default_rng(0)
    customer = Cus_Create()
    for i in range(1000):
        a2 = rates[i]  # federal rate
        a0 = round(a2 + .1 + rng.random() * (.9), 2)  # my rate
        a1 = round(a2 + .1 + rng.random() * (.9), 2)  # opponent rate
        # a0 -> agent1利率, a1 -> agent2利率, a2 -> 最後一筆資料的利率, (b0dis, b1dis, threshold -> cusdata.xlsx對應資料), II -> 4
        ctr, revenue = Rev(a0, a1, a2, customer)
        if i + 1 >= STP and (i + 1) % 10 == 0:
            customer_old[i + 1] = customer.copy()
        hist.append((a0, a1, a2, ctr, revenue))
    hist = np.array(hist)
    pd.DataFrame(hist, columns=['my-rate', 'op-rate',
                 'federal rate', 'copies', 'rev']).astype({"my-rate": float, "op-rate": float, "federal rate": float, "copies": int, "rev": int},
                                                          copy=True).to_excel("p2/history_Rate.xlsx")


def arange(c0, c1, step=0.01):
    return list(map(lambda x: round(x, 2), np.linspace(
        c0, c1, int((c1 - c0) / step) + 1, endpoint=True)))


def Evaluate(h, a0):
    return Rev(a0 + hist[h - 1][2], hist[h - 1][1], hist[h - 1][2], customer_old[h], False)[1]


def Evaluate_B(h, a0):
    return Rev(a0, hist[h - 1][1], hist[h - 1][2], customer_old[h], False)[1]


def MaxMin(h):
    rge = arange(.1, 1, .01)
    arr = [Evaluate(h, a0)
           for a0 in rge]
    x = np.argmax(arr)
    y = np.argmin(arr)
    return (rge[x], arr[x]), (rge[y], arr[y])


Rate_Create()
Hist_Create()
