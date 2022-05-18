
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import collections
import pickle
import matplotlib.gridspec as gridspec
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
    return {1: sigmoid(np.poly1d([-1, 2])(an)), 2: sigmoid(np.poly1d([-3, 3])(an)), 3: sigmoid(np.poly1d([-2, 0])(an))}[tier]


def get_rho(cid, an):
    return rho(customer[cid].tier, an)


def customer_ratio_sensitivity_plot():
    axis = []
    a1 = []
    a2 = []
    a3 = []
    for x in arange(0, 2):
        axis.append(x)
        a1.append(rho(1, x))
        a2.append(rho(2, x))
        a3.append(rho(3, x))
    fig, ax = plt.subplots()
    ax.scatter(axis, a1, c='r', label="tier = 1")
    ax.scatter(axis, a2, c='g', label="tier = 2")
    ax.scatter(axis, a3, c='b', label="tier = 3")
    ax.legend()
    plt.xlabel("Renewal ratio")
    plt.ylabel("Probability of acceptance")
    plt.show()


def customer_best_action_plot():
    axis = []
    a1 = []
    a2 = []
    a3 = []
    for x in arange(0, 2):
        axis.append(x)
        a1.append(rho(1, x)*x)
        a2.append(rho(2, x)*x)
        a3.append(rho(3, x)*x)
    fig, ax = plt.subplots()
    ax.scatter(axis, a1, c='r', label="tier = 1")
    ax.scatter(axis, a2, c='g', label="tier = 2")
    ax.scatter(axis, a3, c='b', label="tier = 3")
    ax.legend()
    plt.xlabel("Renewal ratio")
    plt.ylabel("Probability of acceptance")
    plt.show()


def TierSet(x):
    return {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}[x]


def arange(c0, c1, step=0.01):
    return list(map(lambda x: round(x, 2), np.linspace(
        c0, c1, int((c1 - c0)/step)+1, endpoint=True)))


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
H = 5000
CLUSTER = 250
customer = LoadData()

CONVBOUND = 0
CONVCEIL = 100
EPOCH = 2000
VERSION = "v3sfzdy"
# decayed-episilon-greedy
# ax^2+b, ax + b, a^(2x), 1/(ax^2+b), 1/(ax+b)
ep_mode = 0
epsilon_max = 1
epsilon_min = 0.001
decayed_step = 1000
epFACTOR = 0

learning_rate = 0.4
gamma = 0.6
itl = .4


def SetCentroid(k):
    global CLUSTER
    CLUSTER = int((N2-N) * H / k)


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
                                     deque([customer[i].c0,
                                            customer[i].c1]) for i in range(N, N2)}
        self.__priority_selection_inv = {i:
                                         list(self.__priority_selection[i]) for i in range(N, N2)}
        self.__coeffient = {i: (0, 0) for i in range(N, N2)}
        self.__avoid_set = {i: set() for i in range(N, N2)}
        self.__last_round_for_newdata = 0
        # arange(customer[i].c0, customer[i].c1)[int((len(arange(customer[i].c0, customer[i].c1))-1)/2)],
        # self.__priority_selection = [
        #     deque() for i in range(N, N2)]
        # ax^2+b, ax + b, a^(2x), 1/(ax^2+b), 1/(ax+b)
        if ep_mode == 0:
            self.epFACTOR = (epsilon_min-epsilon_max)/(decayed_step**2-1)
            self.epConstant = epsilon_max - self.epFACTOR
        elif ep_mode == 1:
            self.epFACTOR = (epsilon_min-epsilon_max) / (decayed_step-1)
            self.epConstant = epsilon_max - self.epFACTOR
        elif ep_mode == 2:
            self.epFACTOR = math.exp(math.log(epsilon_min)/(2*decayed_step))
        elif ep_mode == 3:
            self.epFACTOR = (1/epsilon_min-1/epsilon_max)/(decayed_step**2-1)
            self.epConstant = (1/epsilon_max) - self.epFACTOR
        elif ep_mode == 4:
            self.epFACTOR = (1/epsilon_min-1/epsilon_max)/(decayed_step-1)
            self.epConstant = (1/epsilon_max) - self.epFACTOR

        if self.dynamic_graph:
            self.bm = pyqt.Create()
            max, min = th_expected_rev(verbose=False)
            self.bm.addVerticalLine(max, "blue", 1)

    def epsilon_greedy(self, deterministic, actionSet, cid, round, verbose=False, specify=-1, avoidList=None):
        if specify != -1:
            return specify
        cus = customer[cid]
        rv = np.random.uniform(0, 1)
        if deterministic == False:
            if self.epsilon_counter != round:
                self.epsilon_counter = round
                if ep_mode == 0:
                    # ax^2+b
                    self.epsilon = max(self.epFACTOR*round **
                                       2+self.epConstant, epsilon_min)
                elif ep_mode == 1:
                    # ax + b
                    self.epsilon = max(
                        self.epFACTOR * round + self.epConstant, epsilon_min)
                elif ep_mode == 2:
                    # a^(2x)
                    self.epsilon = max(
                        self.epFACTOR ** (2*round), epsilon_min)
                elif ep_mode == 3:
                    # 1/(ax^2+b)
                    self.epsilon = max(
                        1/(self.epFACTOR * round**2 + self.epConstant), epsilon_min)
                elif ep_mode == 4:
                    # 1/(ax+b)
                    self.epsilon = max(
                        1/(self.epFACTOR * round + self.epConstant), epsilon_min)
            self.epsilon = 1
        else:
            rv = 1

        if verbose:
            print(
                f"\nRound_{round}, rv: {rv}, epsilon: {self.epsilon}, random: {rv<self.epsilon}")

        def random_select():
            lst = arange(cus.c0, cus.c1)
            for x in avoidList:
                lst.remove(x)
            an = np.random.choice(lst)
            if verbose:
                print(f"select: {an}")
            return an

        if rv < self.epsilon:
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

    def D(self, x):
        return self.kmeans.predict(self.scaler.transform([x]))[0]

    def alogor1(self, H, N, N2, K, graph=False):
        C = list()
        for i in range(H):
            selected_value = 0
            f1n = 0
            for j in range(N, N2):
                cus = customer[j]
                an = np.random.choice(arange(cus.c0, cus.c1))
                pi = get_rho(j, an)
                selected_value += cus.price * an * pi
                f1n = selected_value
                sn = [f1n]
                C.append(sn)
        C = np.array(C)
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(C)
        norm = scaler.transform(C)
        kmeans = KMeans(n_clusters=K).fit(norm)

        if graph:
            # states visited
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot()
            ax.set_title('States visited')
            ax.scatter(C.T[0], C.T[1], cmap=plt.cm.Set1, s=5)
            ax.set_xlabel("revenue")
            ax.set_ylabel("retention")
            plt.show()

            # draw kmeans plot
            gs = gridspec.GridSpec(2, 2, width_ratios=[
                                   1, 1], height_ratios=[1, 2])
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(gs[0, 0])
            ax.set_title('Original data')
            ax.scatter(norm.T[0], norm.T[1], cmap=plt.cm.Set1, s=5)
            ax.set_xlabel("Revenue")
            ax.set_ylabel("Retention")

            ax = fig.add_subplot(gs[0, 1])
            ax.set_title('Clustering')
            ax.scatter(norm.T[0], norm.T[1],
                       c=kmeans.predict(norm), cmap=plt.cm.Set1, s=5)
            ax.set_xlabel("Revenue")
            ax.set_ylabel("Retention")
            ax = fig.add_subplot(gs[1, :])
            ax.set_title('K-means')
            points = kmeans.cluster_centers_[:, [0, 1]]
            ax.scatter(points[:, 0], points[:, 1], s=5)
            ax.set_xlabel("Revenue")
            ax.set_ylabel("Retention")
            plt.show(block=True)
        return kmeans, scaler

    def GetCurrentCid(self):
        return self.__currentCid

    def Train(self, H, N, N2, learning_rate, gamma, verbose):
        i = 0
        while i <= H:
            i += 1
            self.ROUND += 1
            if SHOW_PROCESS:
                print(self.ROUND)
            selected_value, actionRecorder = 0, []

            f1n = 0
            state = self.D([f1n])
            if i == H:
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
                    #  and self.__priority_selection[j]
                    _an = np.random.choice(arange(0, 2, itl))
                    pi = rho(cus.tier, _an)
                    ask = np.random.binomial(1, pi)
                    self.hist_data[j][_an].append(ask)
                    self.workbook[self.ROUND].append(
                        (_an, ask))
                if self.__priority_selection[j]:
                    xspace = arange(0, 2, itl)
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
                state_p = None
                maxQ = 0

                if j <= N2-2:
                    state_p = self.D([f1n])
                    maxQ = gamma * \
                        self.Q[state_p][self.epsilon_greedy(
                            deterministic=True, actionSet=self.Q[state_p], cid=j+1, round=i, verbose=verbose)]
                oldq = self.Q[state][an]
                self.Q[state][an] = (1 - learning_rate) * self.Q[state][an] + learning_rate * (
                    reward + maxQ)

                # print(f"an: {an} _an: {_an}")
                # print(f"delta: {self.Q[state][an] - oldq}")
                # print(abs(get_rho(j, an)-Objective(an, c0, c1)))
                if self.__priority_selection[j]:
                    if abs(self.Q[state][an] - oldq) <= 0.005:
                        self.__last_round_for_newdata = self.ROUND
                        self.__priority_selection[j].popleft()
                        self.__coeffient[j] = (c0, c1)
                        i = 0
                        if not SILENT:
                            print(f"Round: {self.ROUND}")
                            print(an, Objective(an, c0, c1), get_rho(j, an))
                            print(f"Q: {self.Q[state][an]}")
                            print(self.__priority_selection[j])
                            print("------------------------")

                        # use linear regression to accelerate converge speed
                        if not self.__priority_selection[j]:
                            X = np.array([[x]
                                          for x in self.__priority_selection_inv[j]])
                            Y = np.array([[self.Q[state][s[0]]] for s in X])

                            reg = LinearRegression().fit(X, Y)
                            for x in arange(customer[j].c0, customer[j].c1):
                                # if x not in X:
                                self.Q[state][x] = reg.predict([[x]])[0, 0]
                                if not SILENT:
                                    print(x, self.Q[state][x])
                            if not SILENT:
                                print("------------------------")
                else:
                    if abs(self.Q[state][an] - oldq) <= 0.005:
                        self.__avoid_set[j].add(an)
                        if an not in self.__avoid_set[j]:
                            print(f"{self.ROUND}, an: {an}")
                            print("////////////")

                if verbose:
                    print(
                        f"[an: {an} p: {Objective(xspace, c0, c1)}]")
                if i == H:
                    for an in arange(cus.c0, cus.c1):
                        print(f"[{an}]: {round(self.Q[state][an],3)}")
                    print(".................")
                state = state_p

            if self.dynamic_graph:
                self.bm.updateplot(i, f1n)
            self.c.append(recstate(f1n, actionRecorder))

            # break point
            if len(self.c) >= 2:
                if abs(self.c[-1].rev - self.c[-2].rev) <= CONVBOUND:
                    self.isConv = self.isConv + 1
                else:
                    self.isConv = 0
                if self.isConv == CONVCEIL:
                    print(f"Agent({self.agentId}) Converge")
                    i = H - 1
            if len(self.__avoid_set[j]) == len(arange(cus.c0, cus.c1)) and i < H:
                if not SILENT:
                    print(f"select {an}!!")
                i = H
            yield

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
            f"  Round: {self.ROUND:<5}, Estimate: {self.c[-1].rev:<5}, Real: {get_expected_rev(self.c[-1].actions):<5}")
        print(f"  Choice: {self.c[-1].actions}")
        print(f"  Conv: {self.isConv}")
        print("---------------------------")

    def Generator(self):
        Path(VERSION).mkdir(exist_ok=True)
        FILE_NAME = f"{VERSION}/kmean_N{N}~{N2}H{H}K{CLUSTER}.pickle"
        if not (Path(FILE_NAME).exists() and Path(FILE_NAME).stat().st_size > 0):
            print(f"Kmeans cal..")
            with open(FILE_NAME, 'wb') as f:
                self.kmeans, self.scaler = self.alogor1(H, N, N2, CLUSTER, 0)
                pickle.dump((self.kmeans, self.scaler), f)
                print("Finished")
        else:
            with open(FILE_NAME, 'rb') as f:
                self.kmeans, self.scaler = pickle.load(f)
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
