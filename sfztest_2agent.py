
import pandas as pd
from scipy import rand
from sklearn import cluster
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
import collections
import pickle
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import namedtuple
import Blit
import pyqt

plt.style.use('fast')

cstate = namedtuple('customer', 'price, c0, c1, tier')
recstate = namedtuple('recorder', 'rev action')


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def rho(tier, an):
    return {1: sigmoid(np.poly1d([-9.356, 9.924])(an)), 2: sigmoid(np.poly1d([-10.607, 10.621])(an)), 3: sigmoid(np.poly1d([-11.763, 10.57])(an))}[tier]


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
    ax.plot(axis, a1, c='r', label="tier = 1")
    ax.plot(axis, a2, c='g', label="tier = 2")
    ax.plot(axis, a3, c='b', label="tier = 3")
    ax.legend()
    plt.xlabel("Renewal ratio")
    plt.ylabel("Probability of acceptance")
    plt.show()


def customer_best_action_plot(tier=[1, 1, 1]):
    axis = []
    a1 = []
    a2 = []
    a3 = []
    for x in arange(0, 2, action_step):
        axis.append(x)
        a1.append(rho(1, x)*x)
        a2.append(rho(2, x)*x)
        a3.append(rho(3, x)*x)
    fig, ax = plt.subplots()
    if tier[0]:
        ax.plot(axis, a1, '-o', c='r', label="tier = 1")
    if tier[1]:
        ax.plot(axis, a2, '-o', c='g', label="tier = 2")
    if tier[2]:
        ax.plot(axis, a3, '-o', c='b', label="tier = 3")
    ax.legend()
    plt.xlabel("Renewal ratio")
    plt.ylabel("Value of acceptance * action")
    plt.show()


def TierSet(x):
    return {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}[x]


action_step = 0.01


def arange(c0, c1, step=-1):
    if step == -1:
        step = action_step
    return list(map(lambda x: round(x, 2), np.linspace(
        c0, c1, round((c1 - c0)/step)+1, endpoint=True)))


def LoadData():
    df = pd.read_excel("dataset_2agent.xlsm")
    customer = []
    for i in range(50):
        customer.append(cstate._make((df["price"][i], df["c0"][i],
                        df["c1"][i], df["tier"][i])))
    return customer


def GetInitialPriceSet():
    return customer[N].c0


def GetActionSpace():
    for i in range(N):
        print(arange(customer[i].c0, customer[i].c1))


N = 10
customer = LoadData()
# revenue_total = sum([customer[r].price
#                      for r in range(0, N)])
# retention_total = sum([rho(customer[r].tier, 1)
#                        for r in range(0, N)]) / N

EPOCH = 2000
VERSION = "v3sfzdy"
# decayed-episilon-greedy
# 0 -> power decayed, 1 -> double linear decayed, 2 -> linear decayed, 3 -> 1/x-square, 4 ->  1/x
ep_mode = 0
epsilon_max = 1
epsilon_min = 0.001
decayed_step = 1000
epFACTOR = 0

learning_rate = 0.1
gamma = 0.9

GET_NEW_DATA = True
SHOW_PROCESS = False
SHOW_REV_GRAPH = True
SILENT = False


def th_expected_rev(verbose, start=0):
    v = 0
    h = 0
    maxAct = []
    for i in range(start, N+1):
        cus = customer[i]
        d = [(x, rho(cus.tier, x) * x) for x in arange(cus.c0, cus.c1)]
        d = np.array(d)
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
    for i in range(N):
        cus = customer[i]
        an = series[i]
        pi = rho(cus.tier, an)
        v += pi * cus.price * an
    return v


class sfz_dynamic_model():
    def __init__(self, seed=42, agentId=0, dynamicGraph=False, verbose=False):
        np.random.seed(seed=seed)
        self.agentId = agentId
        self.dynamic_graph = dynamicGraph
        self.verbose = verbose

        self.Q = collections.defaultdict(
            lambda: collections.defaultdict(int))
        self.c = []
        self.ROUND = 0
        self.epsilon_counter = 0
        self.hist_data = collections.defaultdict(
            lambda: collections.defaultdict(list))
        self.workbook = collections.defaultdict(list, {})

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
            max, min = th_expected_rev(verbose=False, start=N-1)
            self.bm.addVerticalLine(max, "blue", 1)

    def epsilon_greedy(self, deterministic, actionSet, cid, round, verbose=False):
        cus = customer[cid]
        rv = np.random.uniform(0, 1)
        if(deterministic == False):
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
        else:
            rv = 1

        if verbose:
            print(
                f"\nRound_{round}, rv: {rv}, epsilon: {self.epsilon}, random: {rv<self.epsilon}")

        def random_select():
            an = np.random.choice(arange(cus.c0, cus.c1))
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
                if not own_dict:
                    return random_select()
                maxid = max(own_dict, key=own_dict.get)
                if verbose:
                    print(f"maxQ: {maxid} {own_dict}")
                return maxid

    def GetCurrentCid(self):
        return self.__currentCid

    def OtherState(self):
        return None

    # Some external factor that may effect customer Willingness

    def ExFactorOnPi(self):
        return 1

    def GetState(self):
        return self.OtherState()

    def GetId(self):
        return self.agentId

    def Train(self, H, N, learning_rate, gamma, verbose):
        for i in range(H):
            if i == H-1:
                self.verbose = True and not SILENT
            self.ROUND += 1
            if SHOW_PROCESS:
                print(self.ROUND)
            actionRecorder = []

            j = N

            an = self.epsilon_greedy(
                False, self.Q[self.GetState()], j, self.ROUND, verbose)
            actionRecorder.append(an)

            cus = customer[j]
            pi = rho(cus.tier, an)
            ask = np.random.binomial(1, pi * self.ExFactorOnPi(an))

            if GET_NEW_DATA:
                self.hist_data[j][self.GetState(), an].append(ask)
                self.workbook[self.ROUND].append(
                    (an, ask))
            e = np.mean(self.hist_data[j][self.GetState(), an]
                        ) if self.hist_data[j][self.GetState(), an] else 0

            reward = cus.price * an * e

            if self.verbose:
                print(self.GetState(), an)

            if self.verbose:
                print(
                    f"Q={self.Q[self.GetState()][an]} reward {reward}")

            statep = None
            if self.GetId() == 0:
                statep = (an, self.GetState()[1])
            else:
                statep = (self.GetState()[0], an)

            maxQ = 0
            if self.Q[statep]:
                own = dict(
                    filter(lambda x: x[0][self.GetId()] == an, self.Q.items()))
                combine = []
                for x in own:
                    combine += own[x].items()

                # if verbose:
                #     print(combine)
                #     print(max(combine))

                maxQ = gamma * max(combine, key=lambda x: x[1])[1]
            self.Q[self.GetState()][an] = (1 - learning_rate) * \
                self.Q[self.GetState()][an] + learning_rate * (reward + maxQ)

            if self.verbose:
                print(
                    f"[an: {an} ask: {ask} p: {e}] pi: {pi}")
                print(f"Q*={self.Q[self.GetState()][an]}")

            if self.dynamic_graph:
                self.bm.updateplot(i, reward)
            self.c.append(recstate(reward, actionRecorder))

            yield an
        #  wait for converge

        # table = dict(sorted(self.Q.items()))
        # for state in table:
        #     # print(state)
        #     for an in self.Q[state]:
        #         def exf(a, b): return 0.5 if a == b else 1 if a < b else 0
        #         e = np.mean(self.hist_data[N][state, an]
        #                     ) if self.hist_data[N][state, an] else 0
        #         if self.GetId() == 0:
        #             reward = customer[N].price * an * e * exf(an, state[1])
        #         else:
        #             reward = customer[N].price * an * e * exf(state[0], an)

        #         while True:
        #             oldq = self.Q[state][an]
        #             self.Q[state][an] = (1 - learning_rate) * \
        #                 self.Q[state][an] + learning_rate * reward
        #             if abs(self.Q[state][an] - oldq) < 0.005:
        #                 break
            # print(self.Q[state].keys())

    def TextResult(self):
        print("---------------------------")
        print(f"Agent({self.agentId})")
        print(
            f"  Round: {self.ROUND:<5}, Estimate: {self.c[-1].rev:<5}, Real: {get_expected_rev(self.c[-1].actions):<5}")
        print(f"  Choice: {self.c[-1].actions}")
        print(f"  Conv: {self.isConv}")
        print("---------------------------")

    def GetQTable(self):
        return self.Q

    def Generator(self):
        Path(VERSION).mkdir(exist_ok=True)
        yield from self.Train(EPOCH, N, learning_rate, gamma, self.verbose)
        maxmin = th_expected_rev(verbose=1)

        if SHOW_REV_GRAPH:
            c = np.array(self.c, dtype=object)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
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
        with Path(f"{VERSION}/histdata{self.agentId}.pickle").open('wb') as f:
            pickle.dump(dict(self.hist_data), f)
        with Path(f"{VERSION}/workbook{self.agentId}.pickle").open('wb') as f:
            pickle.dump(self.workbook, f)

    def LoadHistdata(self):
        with Path(f"{VERSION}/histdata{self.agentId}.pickle").open('rb') as f:
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
        for j in range(N):
            for i in arange(customer[j].c0, customer[j].c1):
                d = hist_data[j][i]
                hist_data[j][i] = np.mean(d) if d else 0 * i
            print("---------------------")
            for i in sorted(hist_data[j].items()):
                print(f"{i[0]:<4}   {i[1]:.2f}")
            print(
                f"max: {sorted(hist_data[j].items(), key=lambda x: x[1], reverse=True)[0]}")

    def GetHistdata(self, cid):
        return self.hist_data[cid]

    def GetFinialChoice(self):
        return self.c[-1].action[0]
