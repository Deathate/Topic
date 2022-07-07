from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import collections
from DataCreator import arange, MaxMin
import DataCreator as dc
import sys
import os
sys.path.insert(1, os.path.abspath('.'))
from scoping import scoping


def A(arr, v=1):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr.reshape(-1, v)


def DictMax(d):
    return (k := max(d, key=d.get), d[k])


class Split:
    def __init__(self, a, b, step):
        self.line = np.linspace(a, b, step, endpoint=True)

    def predict(self, v):
        return np.argmin(np.absolute(self.line - v))


hist = np.array(dc.hist)


def func(x, a, b, c):
    return a * x[:, 0] * x[:, 1] + b + c * x[:, 1]


popt, _ = curve_fit(func, hist[:, [0, 3]], hist[:, 4])
# plt.plot(hist[:, 0]*hist[:, 3], hist[:, 4], 'o')
# plt.plot(hist[:, 0]*hist[:, 3], func(hist[:, [0, 3]], *popt), 'o')
# plt.show()


def M1(H, K=40):
    Q = collections.defaultdict(lambda: collections.defaultdict(int))
    D = []
    rdeltaScalar = MinMaxScaler().fit(A(hist[:, 0] - hist[:, 1]))
    fdeltaScalar = MinMaxScaler().fit(A(hist[:, 0] - hist[:, 2]))

    for i in range(H):
        data = hist[i]
        rdelta = rdeltaScalar.transform(A(data[0] - data[1]))[0, 0]
        fdelta = fdeltaScalar.transform(A(data[0] - data[2]))[0, 0]
        # myrate oprate frate copies rev
        D.append([data[0], data[1], data[2], data[3], data[4], rdelta, fdelta])
    D = np.array(D)

    kmeans_rev = KMeans(n_clusters=K, random_state=0).fit(A(D[:, 3]))
    kmeans_rfdelta = KMeans(n_clusters=K, random_state=0).fit(
        A(D[:, [5, 6]], 2))
    s = -1
    ocopy = 0
    for i in range(H):
        reward = D[i][3] - ocopy
        ocopy = D[i][3]
        state_p = kmeans_rev.predict(A(D[i][3]))[0]
        action = kmeans_rfdelta.predict(A(D[i][[5, 6]], 2))[0]
        maxQ = 0
        if Q[state_p]:
            maxQ = gamma * DictMax(Q[state_p])[1]
        Q[s][action] = (1 - learning_rate) * Q[s][action] + \
            learning_rate * (reward + maxQ)
        # print(f"({s} {action}) = {Q[s][action]}")
        s = state_p

    data = D[H - 1]
    qresult = Q[kmeans_rev.predict(A(data[3]))[0]]
    fresult = {}
    for i in arange(0.1, 1, 0.01):
        rdelta = rdeltaScalar.transform(A(data[2] + i - data[1]))[0, 0]
        fdelta = fdeltaScalar.transform(A(i))[0, 0]
        l = kmeans_rfdelta.predict(A([rdelta, fdelta], 2))[0]
        if l in qresult:
            v = A([i + data[2], qresult[l]], 2)
            fresult[i] = func(v, *popt)
    try:
        m = DictMax(fresult)[0]
    except:
        return M1(H, K - 5)
    ev = dc.Evaluate(H, m)
    maxr = MaxMin(H)[0][1]
    minr = MaxMin(H)[1][1]
    performance = (ev - minr) / (maxr - minr)
    print(performance)
    return performance


# learning_rate = 0.5
# gamma = 0.5
# with scoping():
#     H = 500
#     c = 0
#     for i in range(H, H + 500, 10):
#         c += M1(i)
#     print(c / 50)
