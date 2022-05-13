

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
import time
import scipy.stats as st
from matplotlib.offsetbox import AnchoredText
import matplotlib.gridspec as gridspec
import sys
import matplotlib.style as mplstyle
from pathlib import Path

mplstyle.use('fast')


def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig


# def XB_1(x): return sigmoid(np.poly1d([-1, 2])(x))
# def XB_2(x): return sigmoid(np.poly1d([-3, 3])(x))
# def XB_3(x): return sigmoid(np.poly1d([-2, 0])(x))
def XB_1(x): return .5
def XB_2(x): return .5
def XB_3(x): return .5


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


H = 5000
centroid = 500
Round = 500
N = 2
SAMPLESIZE = 2000
Verbose = 0

np.random.seed(seed=42)
# load dataset
df = pd.read_excel("dataset.xlsm")
customer = []
for i in range(N):
    customer.append((df["price"][i], df["c0"][i], df["c1"][i], df["tier"][i]))
revenue_total = sum([customer[r][0]
                    for r in range(0, N)])
retention_total = sum([rho[customer[r][3]](1)
                       for r in range(0, N)]) / N
# decayed-episilon-greedy
# 0 -> power decayed, 1 -> linear decayed, 2 -> 1/round
ep_mode = 1
epsilon_max = 1
epsilon_min = 0.001
decayed_step = 500
if ep_mode == 0:
    decay = math.exp(math.log(epsilon_min/epsilon_max) /
                     (decayed_step * N))
    epsilon = epsilon_max / decay
elif ep_mode == 1:
    decay = (epsilon_max - epsilon_min) / (decayed_step * N)
    epsilon = epsilon_max

learning_rate = 0.4
gamma = 0.6

Q = {x: collections.defaultdict(int, {}) for x in range(centroid)}


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
            if i % H <= 10:
                an = c0
            elif i % H <= 20:
                an = c1
            elif i % H <= .5 * H:
                an = np.random.choice([c0, c1])
            else:
                an = np.random.choice(np.arange(c0, c1, 0.01))
            an = adjPrec(an)

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

        # draw 3d kmeans plot
        # fig = plt.figure(figsize=(16, 8))
        # ax = fig.add_subplot(121, projection='3d')
        # plt.title('Original data')
        # ax.scatter(norm.T[0], norm.T[1], norm.T[2], cmap=plt.cm.Set1)
        # ax.set_xlabel("Revenue")
        # ax.set_ylabel("Retention")
        # ax.set_zlabel("Price")
        # ax = fig.add_subplot(122, projection='3d')
        # ax.scatter(norm.T[0], norm.T[1], norm.T[2],
        #            c=kmeans.predict(norm), cmap=plt.cm.Set1)
        # ax.set_xlabel("Revenue")
        # ax.set_ylabel("Retention")
        # ax.set_zlabel("Price")
        # plt.show()

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


VERSION = "v2"
if not os.path.isdir(VERSION):
    os.mkdir(VERSION)
FILE_NAME = f"{VERSION}/kmean_N{N}H{H}K{centroid}.pickle"
if not (os.path.isfile(FILE_NAME) and os.path.getsize(FILE_NAME) > 0):
    with open(FILE_NAME, 'wb') as f:
        kmeans, scaler = alogor1(H, N, centroid, False)
        pickle.dump((kmeans, scaler), f)
else:
    with open(FILE_NAME, 'rb') as f:
        kmeans, scaler = pickle.load(f)
# print(kmeans.cluster_centers_)


def epsilon_greedy(deterministic, state, i):
    global Q, epsilon
    rv = np.random.uniform(0, 1)
    if(deterministic == False):
        if ep_mode == 0:
            epsilon = max(epsilon_min, epsilon*decay)
        elif ep_mode == 1:
            epsilon = max(epsilon-decay, epsilon_min)
        elif ep_mode == 2:
            epsilon = 1 / (recorder["round"] + 2)
    else:
        rv = 1
    # if state not in Q:
    #     Q[state] = collections.defaultdict({})
    # return adjPrec(customer[i][1])

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


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@ static_vars(counter=0, conv_ub=collections.defaultdict(int, {}), min_rev=1000000, max_rev=-1, last_conv=0)
def Train(H, N, learning_rate, gamma, recorder):
    if recorder["exp_rev"]:
        Train.min_rev = min(recorder["exp_rev"])
        Train.max_rev = max(recorder["exp_rev"])
    global Q
    if Verbose == 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    elif Verbose == 1:
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.set_xticks([i+1 for i in range(N)])
        ax1.set_ylim(0.5, 1.5)
        ax1.set_title(
            f"optimal selection", fontsize=8)
        hist, = ax1.plot([x+1 for x in range(N)], [x+1 for x in range(N)],
                         linestyle='None', marker='o', markersize=2)
    elif Verbose == 2:
        fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(12, 8))
        ax1.set_xticks([i+1 for i in range(N)])
        ax1.set_ylim(0.5, 1.5)
        ax1.set_title(
            f"optimal selection", fontsize=8)
        hist, = ax1.plot([x+1 for x in range(N)], [x+1 for x in range(N)],
                         linestyle='None', marker='o', markersize=2)
        p_state, = ax2.plot([x+1 for x in range(N)], [x+1 for x in range(N)],
                            linestyle='None', marker='o', markersize=3)
        bar, = ax3.plot([x+1 for x in range(N)], [x+1 for x in range(N)],
                        linestyle='None', marker='o', markersize=2)
        ax2.set_title(
            f"state", fontsize=8)
        ax3.set_title(
            f"pi", fontsize=8)
        ax2.set_ylim(0, centroid)
        ax2.set_xticks([i+1 for i in range(N)])
        ax3.set_xticks([i+1 for i in range(N)])
        ax3.set_ylim(-500, 500)

    plt.subplots_adjust(hspace=0.4)
    ln, = ax.plot([x for x in range(len(recorder["exp_rev"]))],
                  recorder["exp_rev"], marker='o', markersize=2)
    ax.set_title(
        f"{VERSION}_N{N}H{Round}K{centroid}Mode({ep_mode})epsilonM({epsilon_min})step({decayed_step})", fontsize=8)
    fr_round = ax.annotate(
        "0",
        (0, 1),
        xycoords="axes fraction",
        xytext=(10, -20),
        textcoords="offset points",
        ha="left",
        va="top",
        animated=True,
    )
    fr_epsilon = ax.annotate(
        "0",
        (0, 1),
        xycoords="axes fraction",
        xytext=(10, -30),
        textcoords="offset points",
        ha="left",
        va="top",
        animated=True,
    )
    fr_rev = ax.annotate(
        "0",
        (0, 1),
        xycoords="axes fraction",
        xytext=(10, -40),
        textcoords="offset points",
        ha="left",
        va="top",
        animated=True,
    )
    if Verbose == 0:
        bm = Blit.BlitManager(
            fig.canvas, [ln, fr_round, fr_epsilon, fr_rev])
    elif Verbose == 1:
        bm = Blit.BlitManager(
            fig.canvas, [ln, hist, fr_round, fr_epsilon, fr_rev])
    elif Verbose == 2:
        bm = Blit.BlitManager(
            fig.canvas, [ln, hist, fr_round, fr_epsilon, fr_rev, p_state, bar])
    plt.pause(.1)

    def D(x):
        return kmeans.predict(scaler.transform([x]))[0]
    from KeyPress import NonBlockingConsole
    with NonBlockingConsole() as nbc:
        for i in range(recorder["round"]+1, H):
            key = nbc.get_data()
            if key == 'p':
                sys.exit()
            elif key == '\x1b':  # x1b is ESC
                break

            f1n = revenue_total
            f2n = retention_total
            # discretization state(D(f1n, f2n, price), tier)
            state = D([f1n, f2n, customer[0][0]] + TierSet(customer[0][3]))
            selected_value = 0
            selected_p = 0
            actionRecorder = []
            best_actionRecorder = []
            deltaRecorder = []
            stateRecorder = []
            for j in range(0, N):
                # q-learning
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
                    maxQ = gamma * Q[state_p][epsilon_greedy(True, state_p, j)]
                Q[state][an] = (1 - learning_rate) * Q[state][an] + learning_rate * (
                    reward + maxQ)

                best_option = epsilon_greedy(True, state, j)
                best_actionRecorder.append(best_option)
                actionRecorder.append(an)
                deltaRecorder.append(Q[state][best_option])
                stateRecorder.append(state)

                state = state_p

            def expected_rev(series):
                v = 0
                for i in range(N):
                    price = customer[i][0]
                    an = series[i]
                    pi = rho[customer[i][3]](an)
                    v += int(pi * price * an)
                return v
            rev = expected_rev(actionRecorder)
            fr_rev.set_text(f"rev: {rev}")
            if Verbose == 1:
                hist.set_ydata(actionRecorder)
            elif Verbose == 2:
                hist.set_ydata(actionRecorder)
                bar.set_ydata(deltaRecorder)
                p_state.set_ydata(stateRecorder)

            Train.min_rev = min(Train.min_rev, rev)
            Train.max_rev = max(Train.max_rev, rev)
            ax.set_xlim(0, i + 1)
            ax.set_ylim(Train.min_rev - 50, Train.max_rev + 50)
            ln.set_data(np.append(ln.get_xdata(), i),
                        np.append(ln.get_ydata(), rev))

            fr_round.set_text(f"round: {i}")
            fr_epsilon.set_text(f"epsilon: {epsilon}")
            bm.update()
            # plt.pause(0.000001)
            recorder["history"].append([f1n, f2n])
            recorder["exp_rev"].append(rev)
            recorder["actions"].append(actionRecorder)
            recorder["best_actions"].append(best_actionRecorder)
            recorder["round"] = i
            recorder["epsilon"] = epsilon
        recorder["Qtable"] = Q
        plt.show()
    return recorder


RECORDER_FORMAT = f"{VERSION}/recorder_N{N}H{{h}}K{centroid}Mode({ep_mode})epsilonM({epsilon_min})step({decayed_step}).pickle"
FILE_NAME = RECORDER_FORMAT.format(h=Round)

file_recorder = Path(FILE_NAME)
file_recorder.touch(exist_ok=True)
file_address = Path(f"{VERSION}/address.pickle")
file_address.touch(exist_ok=True)

# if file_recorder exists, open, else enter training process
if file_recorder.stat().st_size > 0:
    with file_recorder.open('rb') as f:
        recorder = pickle.load(f)
else:
    file_recorder.unlink()
    # try to find last address
    recorder = {"history": [], "actions": [],
                "round": -1, "exp_rev": [], "best_actions": []}

    LOADER_PATH = RECORDER_FORMAT.format(h=-1)
    with file_address.open('rb') as f:
        if file_address.stat().st_size > 0:
            r = pickle.load(f)
            LOADER_PATH = RECORDER_FORMAT.format(h=r)
            file_loader = Path(LOADER_PATH)
            if file_loader.exists() and file_loader.stat().st_size > 0:
                with file_loader.open('rb') as f:
                    recorder = pickle.load(f)
                    Q = recorder["Qtable"]
                    epsilon = recorder["epsilon"]
            else:
                print(file_loader.name)
                print("Loader not found")
    # training
    recorder = Train(Round, N, learning_rate, gamma, recorder)
    # record train result
    with Path(RECORDER_FORMAT.format(h=recorder["round"])).open('wb') as f:
        pickle.dump(recorder, f)
        # record address
        with file_address.open('wb') as f:
            pickle.dump(recorder["round"], f)


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


def get_expected_rev(series):
    v = 0
    for i in range(N):
        price = customer[i][0]
        an = series[i]
        pi = rho[customer[i][3]](an)
        v += int(pi * price * an)
    return v


def show_rev_result():
    data = np.array(recorder["best_actions"])
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    trev = get_expected_rev(data[-1])
    print(trev)
    threvmx, threvmn = th_expected_rev()
    ax1.axhline(y=624, label="Round 500")
    ax1.axhline(y=630, label="Round 1000")
    ax1.axhline(y=threvmx, color="r")
    ax1.axhline(y=threvmn, color="r")
    ax1.legend(loc=4)
    plt.show()


# show_rev_result()


def best_action_plot(c):
    history = np.array(recorder["best_actions"])
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax1.plot(history[:, c], label="action")
    ax1.set_xlabel("Iteration")
    fig.suptitle("best_action")
    ax1.legend(loc=4)
    plt.show()


# best_action_plot(1)


def performance_plot():
    history = np.array(recorder["history"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history[:, 0]/revenue_total, label="revenue")
    ax1.axvline(x=Round, color="r")
    ax2.plot(history[:, 1]/retention_total, label="retention")
    ax2.axvline(x=Round, color="r")
    ax1.set_xlabel("Iteration")
    ax2.set_xlabel("Iteration")
    fig.suptitle("Performance Result")
    ax1.legend(loc=4)
    ax2.legend(loc=4)
    plt.show()


# performance_plot()


def mean_performance_plot():
    data = normalize(recorder["history"])
    splits = np.array_split(data, 30)
    smean = [(np.mean(x[:, 0]), np.mean(x[:, 1])) for x in splits]
    smean = np.array(smean)
    smean = normalize(smean)
    plt.plot(smean[:, 0], '-o', label="revenue")
    plt.plot(smean[:, 1], '-o', label="retention")
    plt.xlabel("Iteration")
    plt.title("Grouping Performance Result")
    plt.legend()
    plt.show()


# mean_performance_plot()


def sorted_performance_plot():
    plt.rcParams["figure.figsize"] = (6, 6)
    fig, ax = plt.subplots(3, 2)
    data = normalize(recorder["history"])
    total = Round
    segment = 200
    parts = int(total/segment)
    ticks = np.arange(0, parts)

    def draw(i, data):
        def draw_trend(x, y, i, j):
            coef = np.polyfit(x, y, 1)
            trend = np.poly1d(coef)
            ax[i, j].plot(x, trend(x), '--k',
                          label=f"y = {adjPrec(coef[0])}x+{adjPrec(coef[1])}")
        ax[0][i].plot(data[:, 1], label="retention",
                      alpha=0.5+i*0.5, zorder=bool(i))
        ax[0][i].plot(data[:, 0], label="revenue",
                      alpha=1-i*0.5, zorder=not bool(i))
        ax[0][i].set_xlabel("Iteration")
        ax[0][i].title.set_text("Sorted Performance Result(Revenue)")
        ax[0][i].legend(loc=4, frameon=True)
        for x in range(parts+1):
            ax[0, i].axvline(segment*x, color='black', linewidth=.3)

        splits = np.array_split(data, parts)
        smean = [(np.mean(x[:, 0]), np.mean(x[:, 1])) for x in splits]
        smean = np.array(smean)
        smean = normalize(smean)

        ax[1, i].add_artist(AnchoredText(f"{segment} data => 1 marker", loc=2))
        if i == 0:
            draw_trend(ticks, smean[:, 1], 1, i)
        else:
            draw_trend(ticks, smean[:, 0], 1, i)
        ax[1, i].plot(smean[:, 1], '-o',
                      markersize=4)
        ax[1, i].plot(smean[:, 0], '-o',  markersize=4)
        ax[1, i].legend(loc=8)

        if i == 0:
            smean = smean[smean[:, 1].argsort()]
        else:
            smean = smean[smean[:, 0].argsort()]
        ax[2, i].plot(smean[:, 1], '-o',  markersize=4)
        ax[2, i].plot(smean[:, 0], '-o',  markersize=4)
        if i == 0:
            draw_trend(ticks, smean[:, 0], 2, i)
        else:
            draw_trend(ticks, smean[:, 1], 2, i)
        ax[2, i].legend(loc=8)

        # x axis grid
        ax[1, i].set_xticks(ticks, labels=[x if x % 5 == 0 or x ==
                            ticks[-1] else "" for x in ticks])
        ax[1, i].grid(axis='both')

        # y axis grid
        ax[2, i].set_xticks(ticks, labels=[x if x % 5 == 0 or x ==
                            ticks[-1] else "" for x in ticks])
        ax[2, i].grid(axis='both')
    draw(0, data[data[:, 0].argsort()])
    draw(1, data[data[:, 1].argsort()])
    plt.tight_layout()
    plt.show()


# sorted_performance_plot()


# draw revenue, retention distribution
def revenue_retention_dist_plot():
    fig, (ax1, ax2) = plt.subplots(2)
    fig.tight_layout(pad=3.5)

    # revenue
    normalized = preprocessing.scale(np.array(recorder["history"])[:, 0])
    ax1.hist(normalized, density=True, bins=40)
    mn, mx = ax1.get_xlim()
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(normalized)
    ax1.plot(kde_xs, kde.pdf(kde_xs), label="revenue PDF")
    x = np.linspace(-4, 4, 100)
    ax1.plot(x, st.norm.pdf(x, 0, 1), label="standard normal")
    ax1.set_xlabel("z-score")
    ax1.set_ylabel("Probability density")
    ax1.title.set_text('Revenue Distribution')
    ax1.legend()
    # retention
    normalized = preprocessing.scale(np.array(recorder["history"])[:, 1])
    ax2.hist(normalized, density=True, bins=40)
    mn, mx = ax2.get_xlim()
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(normalized)
    ax2.plot(kde_xs, kde.pdf(kde_xs), label="retention PDF")
    x = np.linspace(-4, 4, 100)
    ax2.plot(x, st.norm.pdf(x, 0, 1), label="standard normal")
    ax2.set_xlabel("z-score")
    ax2.set_ylabel("Probability density")
    ax2.title.set_text('Retention Distribution')
    ax2.legend()

    plt.show()


# revenue_retention_dist_plot()


def customer_ratio_sensitivity_plot():
    axis = []
    a1 = []
    a2 = []
    a3 = []
    for x in np.arange(0, 2, 0.05):
        axis.append(x)
        x = adjPrec(x)
        r = sigmoid(rho[1](x))
        a1.append(r)
        r = sigmoid(rho[2](x))
        a2.append(r)
        r = sigmoid(rho[3](x))
        a3.append(r)
    fig, ax = plt.subplots()
    ax.scatter(axis, a1, c='r', label="tier = 1")
    ax.scatter(axis, a2, c='g', label="tier = 2")
    ax.scatter(axis, a3, c='b', label="tier = 3")
    ax.legend()
    plt.xlabel("Renewal ratio")
    plt.ylabel("Probability of acceptance")
    plt.show()


# customer_ratio_sensitivity_plot()
def ActionsPlot(r):
    data = np.array(recorder["actions"])
    fig, ax = plt.subplots()
    ax.plot(data[:, r], linestyle='None', marker='o', markersize=2)
    ax.axvline(x=Round, color='r')
    plt.show()


ActionsPlot(10)
