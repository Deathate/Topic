

from re import T
import Blit
import os.path
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
import collections
import pickle
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import sys
from pathlib import Path

plt.style.use('fast')

np.random.seed(seed=42)


def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig


def XB_1(x): return .5
def XB_2(x): return .5
def XB_3(x): return .5
def XB_1(x): return sigmoid(np.poly1d([-1, 2])(x))
def XB_2(x): return sigmoid(np.poly1d([-3, 3])(x))
def XB_3(x): return sigmoid(np.poly1d([-2, 0])(x))


rho = {1: lambda x: XB_1(x), 2: lambda x:
       XB_2(x), 3: lambda x: XB_3(x)}


def showXB():
    k = 0.9
    print(sigmoid(XB_1(k)))
    k = 1.1
    print(sigmoid(XB_1(k)))
    k = 0.9
    print(sigmoid(XB_2(k)))
    k = 1.1
    print(sigmoid(XB_2(k)))
    k = 0.9
    print(sigmoid(XB_3(k)))
    k = 1.1
    print(sigmoid(XB_3(k)))


def adjPrec(x):
    return round(x, 2)


def normalize(data):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)


def TierSet(x):
    if x == 1:
        return [1, 0, 0]
    elif x == 2:
        return [0, 1, 0]
    elif x == 3:
        return [0, 0, 1]


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def alogor1(H, N, K, graph=False):
    C = list()
    for i in range(0, H):
        select_p = 0
        selected_value = 0
        f1n = revenue_total
        f2n = retention_total
        for j in range(0, N):
            cus = customer[j]
            (price, tier, c0, c1) = (cus[0], cus[3], cus[1], cus[2])
            # if i % H <= 10:
            #     an = c0
            # elif i % H <= 20:
            #     an = c1
            # elif i % H <= .5 * H:
            #     an = np.random.choice([c0, c1])
            # else:
            #     an = np.random.choice(np.arange(c0, c1, 0.01))
            an = np.random.choice(np.arange(c0, c1, 0.01))
            # an = adjPrec(an)

            pi = rho[tier](an)
            selected_value += price * an * \
                round(np.random.binomial(SAMPLESIZE, pi)/SAMPLESIZE, 1)
            select_p += pi
            f1n = selected_value + sum([customer[r][0]
                                        for r in range(j+1, N)])
            f2n = (select_p + sum([rho[customer[r][3]](1)
                                   for r in range(j+1, N)])) / N
            sn = [f1n, f2n, price] + TierSet(tier)
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
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 2])
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


@static_vars(counter=0)
def epsilon_greedy(deterministic, state, i):
    global Q, epsilon
    rv = np.random.uniform(0, 1)
    if(deterministic == False):
        if epsilon_greedy.counter != ROUND:
            epsilon_greedy.counter = ROUND
            if ep_mode == 0:
                epsilon = max(epsilon_min, epsilon*decay)
            elif ep_mode == 1:
                epsilon = max(epsilon-decay, epsilon_min)
            elif ep_mode == 2:
                epsilon = 1 / ROUND ** FRACTION_FACTOR
    else:
        rv = 1

    def random_select():
        cus = customer[i]
        (c0, c1) = (cus[1], cus[2])
        an = np.random.choice(np.arange(c0, c1, 0.01))
        return adjPrec(an)
    if rv < epsilon:
        return random_select()
    else:
        # greedy method
        if len(Q[state]) == 0:
            if(epsilon <= 0.01):
                print("not enough")
            return random_select()
        else:
            return max(Q[state], key=Q[state].get)


@ static_vars(counter=0, conv_ub=collections.defaultdict(int, {}), min_rev=1000000, max_rev=-1, last_conv=0)
def Train(H, N, learning_rate, gamma):
    global Q, ROUND

    def D(x):
        return kmeans.predict(scaler.transform([x]))[0]

    oldfn = 0

    from KeyPress import NonBlockingConsole
    with NonBlockingConsole() as nbc:
        for i in range(H):
            ROUND = i+1
            key = nbc.get_data()
            if key == 'p':
                sys.exit()
            elif key == '\x1b':  # x1b is ESC
                break

            f1n = 0
            f2n = 0
            # discretization state(D(f1n, f2n, price), tier)
            state = None
            selected_value = 0
            selected_p = 0

            def train():
                nonlocal state, selected_value, selected_p, f1n, f2n
                f1n = revenue_total
                f2n = retention_total
                state = D([f1n, f2n, customer[0][0]] + TierSet(customer[0][3]))
                selected_value = 0
                selected_p = 0

                for j in range(0, N):
                    an = epsilon_greedy(False, state, j)
                    (price, tier) = (customer[j][0], customer[j][3])
                    pi = rho[tier](an)
                    selected_value += price * an * \
                        round(np.random.binomial(SAMPLESIZE, pi)/SAMPLESIZE, 1)
                    selected_p += pi

                    old_f1n = f1n
                    f1n = selected_value + sum([customer[r][0]
                                                for r in range(j+1, N)])
                    f2n = (selected_p + sum([rho[customer[r][3]](1)
                                            for r in range(j+1, N)])) / N
                    reward = f1n - old_f1n
                    state_p = None
                    maxQ = 0
                    if j+1 <= N-1:
                        state_p = D([f1n, f2n, customer[j+1][0]] +
                                    TierSet(customer[j+1][3]))
                        maxQ = gamma * \
                            Q[state_p][epsilon_greedy(True, state_p, j)]
                    Q[state][an] = (1 - learning_rate) * Q[state][an] + learning_rate * (
                        reward + maxQ)

                    state = state_p
                return selected_value
            t = train()
            if abs(t - oldfn) <= CONVBOUND:
                isConv = isConv + 1
            else:
                # print(f1n-oldfn)
                isConv = 0
            if isConv == 20 and epsilon <= 0.01:
                print("Converge")
                print(ROUND, train())
                break
            oldfn = t
            c.append(t)
        print(ROUND, c[-1])


def LoadData(N, show):
    df = pd.read_excel("dataset.xlsm")
    customer = []
    for i in range(N):
        customer.append((df["price"][i], df["c0"][i],
                        df["c1"][i], df["tier"][i]))
        if show:
            print(customer[-1])
    return customer


def th_expected_rev():
    v = 0
    h = 0
    for i in range(N):
        cus = customer[i]
        (price, tier, c0, c1) = (cus[0], cus[3], cus[1], cus[2])
        d = [rho[tier](x) * price for x in np.arange(c0, c1, 0.01)]
        v += max(d)
        h += min(d)
    return v, h


N = 10
H = 5000
CENTROID = int(N * H/20)  # 50 > 30 > 20 > 40 > 60
ROUND = 5000
SAMPLESIZE = 2000
VERBOSE = 0
CONVBOUND = 0

customer = LoadData(N, False)
revenue_total = sum([customer[r][0]
                    for r in range(0, N)])
retention_total = sum([rho[customer[r][3]](1)
                       for r in range(0, N)]) / N
# decayed-episilon-greedy
# 0 -> power decayed, 1 -> linear decayed, 2 -> 1/ROUND
ep_mode = 0
epsilon_max = 1
epsilon_min = 0.001
decayed_step = 1000
FRACTION_FACTOR = 0.2
if ep_mode == 0:
    decay = math.exp(math.log(epsilon_min/epsilon_max) / (decayed_step))
    epsilon = epsilon_max / decay
elif ep_mode == 1:
    decay = (epsilon_max - epsilon_min) / (decayed_step)
    epsilon = epsilon_max

learning_rate = 0.4
gamma = 0.6

Q = {x: collections.defaultdict(int, {}) for x in range(CENTROID)}


VERSION = "v3"
if not os.path.isdir(VERSION):
    os.mkdir(VERSION)
FILE_NAME = f"{VERSION}/kmean_N{N}H{H}K{CENTROID}.pickle"
if not (os.path.isfile(FILE_NAME) and os.path.getsize(FILE_NAME) > 0):
    with open(FILE_NAME, 'wb') as f:
        kmeans, scaler = alogor1(H, N, CENTROID, False)
        pickle.dump((kmeans, scaler), f)
else:
    with open(FILE_NAME, 'rb') as f:
        kmeans, scaler = pickle.load(f)

c = []
Train(ROUND, N, learning_rate, gamma)
maxmin = th_expected_rev()
print(maxmin)
plt.plot(c, linewidth=1)
plt.axhline(maxmin[0])
plt.axhline(maxmin[1])
plt.show()
