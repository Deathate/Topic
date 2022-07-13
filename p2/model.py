from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import collections
from DataCreator import arange, MaxMin
import DataCreator as dc
import time
import copy


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

# plt.scatter(D[:, 5], D[:, 6], c=kmeans_rfdelta.predict(A(D[:, [5, 6]], 2)))
# plt.show()
# exit()


hist = dc.hist
D = []
rdeltaScalar = MinMaxScaler().fit(A(hist[:, 0] - hist[:, 1]))
fdeltaScalar = MinMaxScaler().fit(A(hist[:, 0] - hist[:, 2]))

for i in range(1000):
    data = hist[i]
    rdelta = rdeltaScalar.transform(A(data[0] - data[1]))[0, 0]
    fdelta = fdeltaScalar.transform(A(data[0] - data[2]))[0, 0]
    # myrate oprate frate copies rev
    D.append([data[0], data[1], data[2],
              data[3], data[4], rdelta, fdelta])
D = np.array(D)


def func(x, a, b, c):
    return a * x[:, 0] * x[:, 1] + b + c * x[:, 1]


popt, _ = curve_fit(func, hist[:, [0, 3]], hist[:, 4])
# plt.plot(hist[:, 0] * hist[:, 3], hist[:, 4], "o")
# plt.plot(hist[:, 0] * hist[:, 3], func(A(hist[:, [0, 3]], 2), *popt), "o")
# plt.show()
# exit()


class SplitQlearning:
    def __init__(self, k, lr, gamma):
        self.Q = {}
        self.k = k
        self.kmeans_rev = KMeans(
            n_clusters=self.k, random_state=0).fit(A(D[:, 3]))
        self.kmeans_rfdelta = KMeans(n_clusters=self.k, random_state=0).fit(
            A(D[:, [5, 6]], 2))
        self.learning_rate = lr
        self.gamma = gamma
        self.__Qtable()

    def __Qtable(self):
        s = -1
        Q = collections.defaultdict(lambda: collections.defaultdict(int))
        for i in range(1000):
            reward = D[i][3] - (D[i - 1][3] if i >= 1 else 0)
            state_p = self.kmeans_rev.predict(A(D[i][3]))[0]
            action = self.kmeans_rfdelta.predict(A(D[i][[5, 6]], 2))[0]
            maxQ = 0
            if Q[state_p]:
                maxQ = self.gamma * DictMax(Q[state_p])[1]
            Q[s][action] = (1 - self.learning_rate) * Q[s][action] + \
                self.learning_rate * (reward + maxQ)
            s = state_p
            if i + 1 >= 500 and (i + 1) % 10 == 0:
                self.Q[i] = copy.deepcopy(Q)

    def request(self, h):
        data = D[h - 1]
        qresult = self.Q[h - 1][self.kmeans_rev.predict(A(data[3]))[0]]
        fresult = {}
        for i in arange(0.1, 1, 0.01):
            rdelta = rdeltaScalar.transform(A(data[2] + i - data[1]))[0, 0]
            fdelta = fdeltaScalar.transform(A(i))[0, 0]
            l = self.kmeans_rfdelta.predict(A([rdelta, fdelta], 2))[0]
            if l in qresult:
                v = A([i + data[2], qresult[l]], 2)
                fresult[i] = func(v, *popt)
        try:
            m = DictMax(fresult)[0] + data[2]
        except:
            m = -1
        return m


class UnitQlearning:
    def __init__(self, k, lr, gamma):
        self.Q = {}
        self.k = k
        self.kmeans_rfdelta = KMeans(n_clusters=self.k, random_state=0).fit(
            A(D[:, [5, 6]], 2))
        self.learning_rate = lr
        self.gamma = gamma
        self.__Qtable()

    def __Qtable(self):
        Q = collections.defaultdict(int)
        for i in range(1000):
            reward = D[i][3] - (D[i - 1][3] if i >= 1 else 0)
            action = self.kmeans_rfdelta.predict(A(D[i][[5, 6]], 2))[0]
            maxQ = 0
            if Q:
                maxQ = self.gamma * DictMax(Q)[1]
            Q[action] = (1 - self.learning_rate) * Q[action] + \
                self.learning_rate * (reward + maxQ)
            if i + 1 >= 500 and (i + 1) % 10 == 0:
                self.Q[i] = copy.deepcopy(Q)

    def request(self, h):
        data = D[h - 1]
        qresult = self.Q[h - 1]
        fresult = {}
        for i in arange(0.1, 1, 0.01):
            rdelta = rdeltaScalar.transform(A(data[2] + i - data[1]))[0, 0]
            fdelta = fdeltaScalar.transform(A(i))[0, 0]
            l = self.kmeans_rfdelta.predict(A([rdelta, fdelta], 2))[0]
            if l in qresult:
                v = A([i + data[2], qresult[l]], 2)
                fresult[i] = func(v, *popt)
        try:
            m = DictMax(fresult)[0] + data[2]
        except:
            m = -1
        return m


def Compare(rest):
    def GetPerformace(H, m):
        ev = dc.Evaluate_B(H, m)
        performance = MinMaxScaler().fit(
            A([MaxMin(H)[0][1], MaxMin(H)[1][1]])).transform(A(ev))[0, 0]
        return performance
    res1 = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.51, 2.51, 2.54, 2.551629701, 2.578575489, 2.601825868, 2.625630898, 2.648431856, 2.673309461, 2.690037126, 2.692871138, 2.695338762, 2.703052455, 2.704934691,
            2.707438962, 2.705438403, 2.55, 2.56, 2.55, 2.55, 2.54, 2.55, 2.58, 2.564112894, 2.1, 1.1, 1.05, 1.04, 1.05, 1.05, 1.07, 1.09, 1.09, 1.09, 1.1, 1.1, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.08, 1.09, 1]
    res2 = [2.5, 3.4, 3.4, 3.4, 3.4, 3.4, 2.51, 2.51, 2.54, 2.49, 2.48, 2.47, 2.47, 2.52, 3.41, 3.14, 3.13, 2.23, 3.25, 2, 2.9, 1.92, 1.65, 1.65, 2.55, 1.65,
            2.55, 2.55, 1.68, 2.58, 1.19, 0.25, 0.15, 0.15, 0.15, 0.15, 0.16, 0.19, 0.18, 0.19, 0.2, 0.2, 0.19, 0.19, 0.19, 0.29, 1.09, 0.19, 1.08, 1.09, 1.09]
    ctr = 0
    c = []
    for i in range(500, 500 + 500, 10):
        if rest == None:
            rest = res1
        c += [GetPerformace(i, rest[ctr])]
        ctr += 1
    plt.plot(range(len(c)), c, '-o')
    # plt.plot(range(len(c)), MinMaxScaler().fit_transform(
    #     A([dc.rates[x] for x in range(500, 1000, 10)])), '-', alpha=.7, label="federal rate")
    plt.axhline(np.mean(c), color='r', label="mean")
    plt.xlabel("Period")
    plt.ylabel("Performance")
    plt.legend()
    plt.tight_layout()
    # plt.xticks(range(51))
    plt.yticks(arange(0, 1, .2) + [np.mean(c)])
    plt.show()


def M1(H):
    if not hasattr(M1, "QL"):
        M1.QL = [SplitQlearning(x) for x in [50, 40, 30, 20]]
    m = -1
    ql = M1.QL.copy()
    while m == -1 and len(ql) > 0:
        m = ql[0].request(H)
        ql.pop(0)
    if m == -1:
        raise ValueError("m=-1")
    return m


def M2(H):
    if not hasattr(M2, "QL"):
        M2.QL = [UnitQlearning(75, lr=.2, gamma=.5)]
    m = -1
    ql = M2.QL.copy()
    while m == -1 and len(ql) > 0:
        m = ql[0].request(H)
        ql.pop(0)
    if m == -1:
        raise ValueError("m=-1")
    return m


r = [M2(i) for i in range(500, 1000, 10)]
Compare(r)
