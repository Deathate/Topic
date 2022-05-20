
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import collections
import pickle
from pathlib import Path
from collections import namedtuple
import pyqt
from scipy.optimize import curve_fit
from collections import deque

plt.style.use('fast')

cstate = namedtuple('customer', 'price, c0, c1, tier')
recstate = namedtuple('recorder', 'rev actions')


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def rho(tier, an):
    return {1: sigmoid(np.poly1d([-9.356, 9.924])(an)), 2: sigmoid(np.poly1d([-10.607, 10.621])(an)), 3: sigmoid(np.poly1d([-11.763, 10.57])(an))}[tier]


def get_rho(cid, an):
    return rho(customer[cid].tier, an)


def customer_ratio_sensitivity_plot(r0=0, r1=2):
    axis = []
    a1 = []
    a2 = []
    a3 = []
    for x in arange(r0, r1):
        axis.append(x)
        a1.append(rho(1, x))
        a2.append(rho(2, x))
        a3.append(rho(3, x))
    fig, ax = plt.subplots()
    ax.plot(axis, a1, c='r', label="tier = 1")
    ax.plot(axis, a2, c='g', label="tier = 2")
    ax.plot(axis, a3, c='b', label="tier = 3")
    ax.legend()
    plt.xlabel("Renewal ratio")
    plt.ylabel("Probability of acceptance")
    plt.show()


def customer_best_action_plot():
    axis = []
    a1 = []
    a2 = []
    a3 = []
    for x in arange(.7, 1.3):
        axis.append(x)
        a1.append(rho(1, x)*x)
        a2.append(rho(2, x)*x)
        a3.append(rho(3, x)*x)
    fig, ax = plt.subplots()
    ax.plot(axis, a1, c='r', label="tier = 1")
    ax.plot(axis, a2, c='g', label="tier = 2")
    ax.plot(axis, a3, c='b', label="tier = 3")
    ax.legend()
    plt.xlabel("Renewal ratio")
    plt.ylabel("Expected renewal ratio")
    plt.show()


def TierSet(x):
    return {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}[x]


def arange(c0, c1, step=0.01, count=-1):
    if count == -1:
        count = round((c1 - c0)/step)+1
    return list(map(lambda x: round(x, 2), np.linspace(
        c0, c1, count, endpoint=True)))


def normalize(data):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


def LoadData():
    df = pd.read_excel("dataset.xlsm")
    customer = []
    for i in range(50):
        customer.append(cstate._make((df["price"][i], df["c0"][i],
                        df["c1"][i], df["tier"][i])))
    return customer


def ShowCusData(i):
    print(customer[i])


def GetInitialPriceSet():
    return [customer[i].c0 for i in range(N, N2)]


def GetActionSpace():
    return [arange(customer[i].c0, customer[i].c1) for i in range(N, N2)]


GET_NEW_DATA = True
SHOW_PROCESS = False
SHOW_REV_GRAPH = True
SILENT = False
N = 0
N2 = 10
customer = LoadData()


EPOCH = 2000
VERSION = "v3sfzdy"


learning_rate = 0.4
gamma = 0.9
itl = 2


def th_expected_rev(verbose):
    v = 0
    h = 0
    maxAct = []
    for i in range(N, N2):
        cus = customer[i]
        d = np.array([(x, rho(cus.tier, x) * x)
                     for x in arange(cus.c0, cus.c1)])
        d = d[np.argsort(d[:, 1])]
        v += d[-1][1] * cus.price
        maxAct.append(d[-1][0])
        h += d[0][1] * cus.price
    if verbose:
        print(f"th_best: {maxAct}")
        print(f"thmax: {v}, th_min: {h}")
    return v, h


def get_expected_rev(series):
    v = 0
    for i in range(N, N2):
        cus = customer[i]
        an = series[i-N]
        pi = rho(cus.tier, an)
        v += pi * cus.price * an
    return v


def Objective(x, a, b):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    return sigmoid(np.poly1d([a, b])(x))


def FitCurve(x, y):
    coeffient, _ = curve_fit(Objective, x, y)
    return coeffient


class sfz_dynamic_model():
    def __init__(self, seed=42, agentId=0, dynamicGraph=False, verbose=False):

        np.random.seed(seed=seed)
        self.agentId = agentId
        self.dynamic_graph = dynamicGraph
        self.verbose = verbose

        self.Q = collections.defaultdict(
            lambda: collections.defaultdict(int))
        self.c = []
        self.isConv = 0
        self.ROUND = 0
        self.epsilon_counter = 0
        self.hist_data = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.workbook = collections.defaultdict(list, {})
        self.kmeans = None
        self.scaler = None
        self.__currentCid = -1

        self.__priority_selection = {i:
                                     deque(arange(customer[i].c0, customer[i].c1, count=3)) for i in range(N, N2)}
        self.__priority_selection_inv = {i:
                                         list(self.__priority_selection[i]) for i in range(N, N2)}
        self.__coeffient = {i: (0, 0) for i in range(N, N2)}
        self.__avoid_set = {i: set() for i in range(N, N2)}
        self.__last_round_for_newdata = 0

        if self.dynamic_graph:
            self.bm = pyqt.Create()
            max, min = th_expected_rev(verbose=False)
            self.bm.addVerticalLine(max, "blue", 1)

    def epsilon_greedy(self, deterministic, actionSet, cid, round, verbose=False, specify=-1, avoidList=None):
        if specify != -1:
            return specify
        cus = customer[cid]

        def random_select():
            lst = arange(cus.c0, cus.c1)
            for x in avoidList:
                lst.remove(x)
            an = np.random.choice(lst)
            if verbose:
                print(f"select: {an}")
            return an

        if deterministic == False:
            return random_select()
        else:
            # greedy method
            if len(actionSet) == 0:
                if self.epsilon <= 0.01 and not deterministic:
                    print("not enough")
                return random_select()
            else:
                own_dict = {k: actionSet[k]
                            for k in actionSet.keys()}
                own_dict = dict(
                    filter(lambda elem: elem[0] >= cus.c0 and elem[0] <= cus.c1, own_dict.items()))
                if not own_dict:
                    return random_select()
                maxid = max(own_dict, key=own_dict.get)
                if verbose:
                    print(
                        f"maxQ: {maxid}")
                    for x, y in sorted(own_dict.items()):
                        print(x, y)
                    if deterministic:
                        print("**")
                return max(own_dict, key=own_dict.get)

    def Train(self, H, N, N2, learning_rate, gamma, verbose):
        i = 0
        while i <= H:
            i += 1
            self.ROUND += 1
            if SHOW_PROCESS:
                print(self.ROUND)
            selected_value, actionRecorder = 0, []

            f1n = 0
            state = 0
            if i == H+1:
                verbose = True and not SILENT
            for j in range(N, N2):
                self.__currentCid = j

                cus = customer[j]

                if not self.__priority_selection[j]:
                    specify = -1
                else:
                    specify = self.__priority_selection[j][0]
                # if i = H, than select the best one
                an = self.epsilon_greedy(
                    i == H+1, self.Q[state], j, round=i, verbose=verbose, specify=specify, avoidList=self.__avoid_set[j])
                yield an
                actionRecorder.append(an)

                if GET_NEW_DATA and self.__priority_selection[j]:
                    self.__last_round_for_newdata += 1
                    #  and self.__priority_selection[j]
                    _an = np.random.choice(self.SelectionSpace(j))
                    pi = rho(cus.tier, _an)
                    ask = np.random.binomial(1, pi)
                    self.hist_data[j][_an].append(ask)
                    self.workbook[self.ROUND].append(
                        (_an, ask))
                if self.__priority_selection[j]:
                    xspace = self.SelectionSpace(j)
                    e = [np.mean(self.hist_data[j][x])
                         for x in xspace]
                    c0, c1 = FitCurve(xspace, e)
                else:
                    c0, c1 = self.__coeffient[j]
                pi = Objective(an, c0, c1)
                selected_value += cus.price * an * pi

                oldfn = f1n
                f1n = selected_value

                reward = f1n - oldfn

                oldq = self.Q[state][an]
                self.Q[state][an] = (1 - learning_rate) * \
                    self.Q[state][an] + learning_rate * reward

                if self.__priority_selection[j]:
                    if abs(self.Q[state][an] - oldq) <= .01:
                        self.__priority_selection[j].popleft()
                        self.__coeffient[j] = (c0, c1)
                        # self.FitGraph(c0, c1)
                        i = 0
                        if not SILENT:
                            print(f"Round: {self.ROUND}")
                            print(an, Objective(an, c0, c1), get_rho(j, an))
                            print(f"Q: {self.Q[state][an]}")
                            print(self.__priority_selection[j])
                            print("------------------------")
                        # self.FitGraph(c0, c1)
                        # use linear regression to accelerate converge speed
                        # if not self.__priority_selection[j]:
                        #     X = np.array([[x]
                        #                   for x in self.__priority_selection_inv[j]])
                        #     Y = np.array([[self.Q[state][s[0]]] for s in X])

                        #     reg = LinearRegression().fit(X, Y)
                        #     for x in arange(customer[j].c0, customer[j].c1):
                        #         # if x not in X:
                        #         self.Q[state][x] = reg.predict([[x]])[0, 0]
                        #         if not SILENT:
                        #             print(x, self.Q[state][x])
                        #     if not SILENT:
                        #         print("------------------------")
                else:
                    if abs(self.Q[state][an] - oldq) <= 0.005:
                        if an not in self.__avoid_set[j]:
                            if not SILENT:
                                print(f"{self.ROUND}, an: {an}")
                                print("////////////")
                        self.__avoid_set[j].add(an)

                if verbose:
                    print(
                        f"[an: {an} p: {Objective(xspace, c0, c1)}]")

            if self.dynamic_graph:
                self.bm.updateplot(i, f1n)
            self.c.append(recstate(f1n, actionRecorder))

            if len(self.__avoid_set[j]) == len(arange(cus.c0, cus.c1)) and i < H:
                i = H
            yield

    def SelectionSpace(self, cid):
        return arange(0, 2, count=itl)

    def FitGraph(self, c0, c1):
        fd = arange(customer[0].c0, customer[0].c1)
        d = [get_rho(0, r) for r in fd]
        plt.plot(fd, d, color='red')
        plt.plot(fd, Objective(fd, c0, c1), '--', color='b')
        plt.show()

    def GetFinalChoice(self):
        return self.c[-1].actions

    def GetFinalRound(self):
        return self.__last_round_for_newdata

    def TextResult(self):
        print("-----------Text Result----------------")
        print(f"Agent({self.agentId})")
        print(
            f"  Round: {self.ROUND:<5}, Estimate: {round(self.c[-1].rev,2)}, Real: {round(get_expected_rev(self.c[-1].actions),2)}, Data Amount: {self.__last_round_for_newdata}")
        print(f"  Choice: {self.c[-1].actions}")
        print(f"  Conv: {self.isConv}")
        print("---------------------------")

    def Generator(self):
        Path(VERSION).mkdir(exist_ok=True)
        yield from self.Train(EPOCH, N, N2, learning_rate, gamma, self.verbose)
        print("---------------------")
        maxmin = th_expected_rev(verbose=not SILENT)

        if SHOW_REV_GRAPH:
            c = np.array(self.c, dtype=object)
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.set_title("revenue known by agent")
            ax2.set_title("revenue known by god")
            ax1.plot(c[:, 0], linewidth=1)
            ax1.axhline(maxmin[0], color="r")
            ax1.axhline(maxmin[1], color="r")
            ax2.plot(np.array([get_expected_rev(x)
                               for x in c[:, 1]]), linewidth=1)
            ax2.axhline(maxmin[0], color="r")
            ax2.axhline(maxmin[1], color="r")
            plt.show()

    def SavePickle(self):
        with Path(f"{VERSION}/histdata{self.agentId}_{N}_{N2}.pickle").open('wb') as f:
            pickle.dump(dict(self.hist_data), f)
        with Path(f"{VERSION}/workbook{self.agentId}_{N}_{N2}.pickle").open('wb') as f:
            pickle.dump(self.workbook, f)

    def LoadHistdata(self):
        with Path(f"{VERSION}/histdata{self.agentId}_{N}_{N2}.pickle").open('rb') as f:
            return pickle.load(f)

    def SaveWorkbook(self):
        workbook = self.workbook
        with pd.ExcelWriter(f'{VERSION}/histdata.xlsx') as writer:
            row = 0
            for i in range(1, max(workbook.keys())+1):
                pd.DataFrame([[f"第{i}期"]]).to_excel(
                    writer, startrow=row, index=0, header=0)
                ratios = ["renewal ratio"] + \
                    np.array(workbook[i])[:, 0].tolist()
                asks = ["result"] + list(
                    map(lambda x: "Y" if x else "N",
                        np.array(workbook[i])[:, 1].tolist()))
                pd.DataFrame([ratios]).to_excel(
                    writer, startrow=row+1, index=0, header=0)
                pd.DataFrame([asks]).to_excel(
                    writer, startrow=row+2, index=0, header=0)
                row += 3
            pd.DataFrame([[f"客戶 {r}" for r in range(1, N+1)]]).to_excel(
                writer, startrow=0, startcol=1, index=0, header=0)

    def AssignHistdata(self, hist):
        self.hist_data = hist

    def ShowHistdata(self):
        hist_data = self.hist_data
        for j in range(N, N2):
            for i in arange(customer[j].c0, customer[j].c1):
                d = hist_data[j][i]
                hist_data[j][i] = (np.mean(d) if d else 0) * i
            print("---------------------")
            for i in sorted(hist_data[j].items()):
                print(f"{i[0]:<4}   {i[1]:.2f}")
            print(
                f"max: {sorted(hist_data[j].items(), key=lambda x: x[1], reverse=True)[0]}")

    def GetHistdata(self, cid):
        return self.hist_data[cid]
