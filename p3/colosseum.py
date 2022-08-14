
from itertools import chain
from collections import namedtuple
import DataCenter as dc
from DataCenter import Starter
import numpy as np
import itertools
from itertools import combinations
import time
import multiprocessing as mp
from multiprocessing import Process
from multiprocess import Pool
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pandas as pd
import sys
import warnings
from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
import collections
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import math
import copy

warnings.simplefilter("ignore")


def A(arr, v=1):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr.reshape(-1, v)


def static(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def getrng():
    create_rng.ctr += 1
    return np.random.default_rng(create_rng.ctr)


@static(ctr=0)
def create_rng(func):
    if not hasattr(func, "rng"):
        func.rng = getrng()
    return func


def FilterDataAll(l, **data):
    if len(data["hist"]) < -l:
        raise RuntimeError()
    state = namedtuple("d", "mrate, orate, mcopy, ocopy, mrev, orev, rate")
    d = []
    if data["id"] == 0:
        d = [state._make((hist.a0, hist.a1, hist.copy0, hist.copy1,
                          hist.rev0, hist.rev1, hist.rate)) for hist in data["hist"][l:]]
    elif data["id"] == 1:
        d = [state._make((hist.a1, hist.a0, hist.copy1, hist.copy0,
                          hist.rev1, hist.rev0, hist.rate)) for hist in data["hist"][l:]]
    return d


def Middle(**data):
    return data["currate"] + dc.setting.II / 2


def Max(**data):
    return data["currate"] + dc.setting.II


def Min(**data):
    return data["currate"]


@create_rng
def Random(rng=None, **data):
    if not rng:
        return data["currate"] + Random.rng.uniform(0, dc.setting.II)
    else:
        return data["currate"] + rng.uniform(0, dc.setting.II)


@create_rng
def OponentBase(**data):
    hist = data["hist"]
    if not hist:
        return Random(OponentBase.rng, **data)
    else:
        fd = FilterDataAll(-1, **data)[0]
        return fd.orate + .1


@create_rng
def OponentBaseDynamic(**data):
    hist = data["hist"]
    if not hist:
        return Random(OponentBaseDynamic.rng, **data)
    else:
        fd = FilterDataAll(-1, **data)[0]
        if fd.mrev < fd.orev:
            if fd.mcopy < fd.ocopy:
                return fd.orate + .3
            else:
                return fd.orate - .3
        else:
            if fd.mrev > 0:
                if fd.orev > 0:
                    return fd.mrate
                else:
                    return fd.mrate
            else:
                if fd.orev > 0:
                    return fd.orate+.2
                else:
                    return data["currate"]


# def Chen1(**data):
#     hist = data["hist"]
#     if not hist:
#         return Random(**data)
#     else:
#         fd = FilterDataAll(-20, **data)
#         m = np.mean([x.mrate for x in fd])
#         return m


# def Chen2(**data):
#     hist = data["hist"]
#     if not hist:
#         return Random(**data)
#     else:
#         if data["id"] == 0:
#             myrates = [x.a0 for x in hist]
#         elif data["id"] == 1:
#             myrates = [x.a1 for x in hist]
#         m = np.sum(myrates) * 0.5
#         return m

@create_rng
def Chen3(**data):
    hist = data["hist"]
    if not hist:
        return Random(Chen3.rng, **data)
    else:
        hist = hist[-1]
        m = (hist.a0 + hist.a1) / 2
        return m


@create_rng
def ES(**data):
    hist = data["hist"]
    if not hist or len(hist) <= 5:
        return Random(ES.rng, **data)
    else:
        fd = FilterDataAll(-5, **data)
        st = np.mean([x.mrate for x in fd[:-1]])
        yt = fd[-1].mrate
        alpha = .9
        m = alpha * yt + (1-alpha) * st
        return m


@create_rng
def ME(**data):
    hist = data["hist"]
    if not hist or len(hist) <= 5:
        return Random(ME.rng, **data)
    else:
        fd = FilterDataAll(-5, **data)
        m = np.mean([x.orate for x in fd])
        return m


@create_rng
def Chen5(**data):
    hist = data["hist"]
    if not hist or len(hist) <= 20:
        return Random(Chen5.rng, **data)
    else:
        fd = FilterDataAll(-20, **data)
        m = np.mean([x.orate for x in fd]) + .2
        return m


@create_rng
def Chen6(**data):
    hist = data["hist"]
    if not hist:
        return Random(Chen6.rng, **data)
    else:
        hist = hist[-1]
        m = (hist.a0 + hist.a1) / 2
        return m + .2


@create_rng
def Chiang1(**data):
    hist = data["hist"]
    if not hist or len(hist) <= 10:
        return Random(Chiang1.rng, **data)
    else:
        fd = FilterDataAll(0, **data)
        X = A([x.orate for x in fd])
        Y = A([x.mrate for x in fd])
        ft = np.mean([x.orate for x in fd[-10:]]) + .1
        reg = LinearRegression().fit(X, Y)
        m = reg.predict(A(ft))[0, 0]
        return m


@create_rng
def Chiang2(**data):
    hist = data["hist"]
    if not hist or len(hist) <= 10:
        return Random(Chiang2.rng, **data)
    else:
        fd = FilterDataAll(0, **data)
        X1 = A([x.orate for x in fd])
        X2 = A(data["rates"])
        X = np.hstack((X1, X2))
        Y = A([x.mrate for x in fd])
        ft = [np.mean([x.orate for x in fd[-10:]]) + .1, data["currate"]]
        reg = LinearRegression().fit(X, Y)
        m = reg.predict(A(ft, 2))[0, 0]
        return m


class Split:
    def __init__(self, a, b, step):
        self.line = np.linspace(a, b, step, endpoint=True)

    def predict(self, v):
        return np.argmin(np.absolute(self.line - v))

    def map(self, v):
        return self.line[v]

    def __len__(self):
        return len(self.line)


class BanditProblem:
    def __init__(self, upb):
        self.Q = collections.defaultdict(int)
        self.upb = upb
        self.rdeltaScalar = Split(0, upb, int(upb/.2) + 1)
        self.fdeltaScalar = Split(0, upb, int(upb/.5) + 1)

    def Fit(self, fd):
        self.fd = fd
        d = fd[-1]
        reward = d.mrev - d.orev
        rdelta = self.rdeltaScalar.predict(d.mrate - d.orate)
        fdelta = self.fdeltaScalar.predict(d.mrate - d.rate)
        action = (rdelta, fdelta)
        self.Q[action] = reward
        return self

    def DictMax(self, d):
        return (k := max(d, key=d.get), d[k])

    def predict(self, nrate):
        data = self.fd[-1]
        fresult = {}
        for i in dc.arange(0, self.upb, 0.1):
            rdelta = self.rdeltaScalar.predict(nrate + i - data.orate)
            fdelta = self.fdeltaScalar.predict(i)
            l = (rdelta, fdelta)
            if l in self.Q:
                fresult[i] = self.Q[l]
        if fresult:
            qstar = self.DictMax(fresult)
            if qstar[1] > 0:
                return qstar[0] + nrate
        return None


@create_rng
def Bandit(**data):
    if not hasattr(Bandit, "bp"):
        Bandit.bp = BanditProblem(dc.setting.II)
    hist = data["hist"]
    if not hist:
        return Random(Bandit.rng, **data)
    else:
        fd = FilterDataAll(0, **data)
        m = Bandit.bp.Fit(fd).predict(data["currate"])
        if m:
            return m
        else:
            return Random(**data)


class QLearningProblem:
    def __init__(self, upb, lr=1, gamma=.4):
        self.learning_rate = lr
        self.gamma = gamma
        self.ofdeltaScalar = Split(0, upb, int(upb / .5) + 1)
        self.actScalar = Split(-2, 2, 10)
        self.Q = collections.defaultdict(
            lambda: {i: 0 for i in range(len(self.actScalar))})
        self.Nt = collections.defaultdict(
            lambda: collections.defaultdict(lambda: 0))
        self.ctr = 0
        self.exploration = 1
        self.rng = getrng()

    def State(self, d):
        return d.rate, self.ofdeltaScalar.predict(d.orate - d.rate)

    def Fit(self, fd):
        self.fd = fd
        d = fd[-1]
        reward = d.mrev - d.orev
        s = self.State(d)
        action = self.actScalar.predict(d.mrate - d.orate)
        self.Nt[s][action] += 1
        maxQ = self.gamma * self.DictMax(self.Q[s])[1] if self.Q[s] else 0
        self.Q[s][action] = (1 - self.learning_rate) * self.Q[s][action] + \
            self.learning_rate * (reward + maxQ)
        self.ctr += 1
        return self

    def DictMax(self, d):
        return (k := max(d, key=d.get), d[k])

    def DictMin(self, d):
        return (k := min(d, key=d.get), d[k])

    def Uncertainty(self, timestep, nt):
        if nt == 0:
            return sys.float_info.max
        return math.sqrt(math.log(timestep)/nt)

    def predict(self, nrate):
        d = self.fd[-1]
        s = self.State(d)

        m = self.DictMax(self.Q[s])
        if m[1] > 20000:
            return self.actScalar.map(m[0]) + d.orate

        if self.rng.random() < .35:
            scalar = MinMaxScaler().fit(
                A([self.DictMax(self.Q[s])[1], self.DictMin(self.Q[s])[1]]))
            lsm = self.DictMax(
                {x: scalar.transform(A(self.Q[s][x]))[0, 0] + self.exploration * self.Uncertainty(self.ctr, self.Nt[s][x]) for x in self.Q[s]})
            return self.actScalar.map(lsm[0]) + d.orate
        else:
            return self.actScalar.map(self.rng.integers(len(self.actScalar))) + d.orate


@create_rng
def QLearning(**data):
    if not hasattr(QLearning, "bp"):
        QLearning.bp = QLearningProblem(dc.setting.II)
    hist = data["hist"]
    if not hist:
        return Random(QLearning.rng, **data)
    else:
        fd = FilterDataAll(0, **data)
        m = QLearning.bp.Fit(fd).predict(data["currate"])
        if m:
            return m
        else:
            return Random(**data)


class LinearSoftmaxAgent(object):
    def __init__(self, state_size, action_size):
        self.rng = getrng()
        self.state_size = state_size
        self.action_size = action_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.theta = self.rng.random(state_size * action_size)
        self.alpha = .01
        self.gamma = .99

    def hasState(self):
        return self.states

    def store(self, state, action):
        self.states.append(state)
        self.actions.append(action)

    def storeReward(self, reward):
        self.rewards.append(reward)

    def _phi(self, s, a):
        encoded = np.zeros([self.action_size, self.state_size])
        encoded[a] = s
        return encoded.flatten()

    def pi(self, s):
        z = [self.theta.dot(self._phi(s, a)) /
             1000 for a in range(self.action_size)]
        weights = np.array([np.exp(z[a])
                           for a in range(self.action_size)])
        return weights / np.sum(weights)

    def act(self, state):
        probs = self.pi(state)
        a = self.rng.choice(range(0, self.action_size), p=probs)
        return (a, probs[a])

    def _gradient(self, s, a):
        expected = 0
        probs = self.pi(s)
        for b in range(0, self.action_size):
            expected += probs[b] * self._phi(s, b)
        return self._phi(s, a) - expected

    def _R(self, t):
        total = 0
        for tau in range(t, len(self.rewards)):
            total += self.gamma**(tau - t) * self.rewards[tau]
        return total

    def train(self):
        self.rewards = preprocessing.scale(
            A(self.rewards)).flatten()
        for t in range(len(self.states)):
            s = self.states[t]
            a = self.actions[t]
            r = self._R(t)
            grad = self._gradient(s, a)
            self.theta += self.alpha * r * grad
        self.states = []
        self.actions = []
        self.rewards = []


@ create_rng
def PolicyGradient(**data):
    if not hasattr(PolicyGradient, "bp"):
        PolicyGradient.stateScalar = Split(
            0, dc.setting.II, int(dc.setting.II / .5) + 1)
        PolicyGradient.actScalar = Split(
            0, dc.setting.II, int(dc.setting.II / 1) + 1)
        PolicyGradient.bp = LinearSoftmaxAgent(
            len(PolicyGradient.stateScalar), len(PolicyGradient.actScalar))
    hist = data["hist"]
    if not hist:
        return Random(PolicyGradient.rng, **data)
    else:
        fd = FilterDataAll(0, **data)[-1]
        if PolicyGradient.bp.hasState():
            reward = fd.mrev - fd.orev
            PolicyGradient.bp.storeReward(reward)
            if data["timestep"] % 20 == 0:
                PolicyGradient.bp.train()

        state = PolicyGradient.stateScalar.predict(fd.rate)
        action, prob = PolicyGradient.bp.act(state)

        PolicyGradient.bp.store(state, action)
        return PolicyGradient.actScalar.map(action) + data["currate"]
################################################################################################################


def OneOneContest(queue, sema, ma, mb, show=False):
    rates = dc.rates
    hist = Starter()
    for i in range(1000):
        a0 = ma(currate=rates[i], id=0, hist=hist,
                rates=rates[:i], timestep=i+1)
        a1 = mb(currate=rates[i], id=1, hist=hist,
                rates=rates[:i], timestep=i+1)
        a0 = np.clip(a0, rates[i], rates[i] + dc.setting.II)
        a1 = np.clip(a1, rates[i], rates[i] + dc.setting.II)

        hist.append(dc.Rev(i, a0, a1))
    pd.DataFrame(hist, columns=[ma.__name__, mb.__name__,
                 'copy0', 'copy1', 'rev0', 'rev1', 'rate']).to_excel("p3/hist.xlsx")
    queue.put(((ma.__name__, mb.__name__), dc.Result(
        hist, "_".join([ma.__name__, mb.__name__]), show=show)))
    sema.release()


@static(queue=mp.Queue(), sema=mp.Semaphore(20))
def NNContest(battleList):
    dc.ClearGraph()
    with open("p3/report.txt", "w") as f:
        print("".ljust(40, "-"))
        maxLength = max(
            map(lambda x: len(x), [x.__name__ for x in battleList])) + 2
        comb = list(combinations(battleList, 2))
        results = []
        if len(comb) > 1:
            for job in tqdm(comb):
                NNContest.sema.acquire()
                p1 = Process(target=OneOneContest, args=(
                    NNContest.queue, NNContest.sema, *job,))
                p1.start()
        else:
            OneOneContest(NNContest.queue, NNContest.sema, *comb[0])
        results = [NNContest.queue.get() for _ in comb]

        result = {x.__name__: 0 for x in battleList}
        table = {}
        for (ma, mb), x in results:
            result[ma] += (x == 1 or x == 3)
            result[mb] += (x == 2 or x == 3)
            txt = [int(x == 1 or x == 3), int(x == 2 or x == 3)]
            print(
                f"{ma:<{maxLength}} vs {mb:>{maxLength}}" + f"  --  {txt[0]}: {txt[1]}")
            print(
                f"{ma:<{maxLength}} vs {mb:>{maxLength}}" + f"  --  {txt[0]}: {txt[1]}", file=f)
            table[(ma, mb)] = (x == 1 or x == 3)
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        print("".ljust(40, "-"))
        print(result)
        print(result, file=f)

    with open("result.pickle", "wb") as f:
        pickle.dump((result, table), f)


def GShow():
    with open("result.pickle", "rb") as f:
        result, table = pickle.load(f)
    namelist = [x[0] for x in result]
    cellColours = []
    for i in range(len(namelist)):
        cellColours.append([])
        for j in range(len(namelist)):
            if i == j:
                cellColours[-1].append("#2C3E50")
            else:
                if (namelist[i], namelist[j]) in table:
                    cellColours[-1].append(
                        "#56b5fd" if table[namelist[i], namelist[j]] else "#EC7063")
                elif (namelist[j], namelist[i]) in table:
                    cellColours[-1].append(
                        "#EC7063" if table[namelist[j], namelist[i]] else "#56b5fd")
                else:
                    cellColours[-1].append("#F9E79F")

    fig, ax = plt.subplots(1, 1)
    namelist = list(map(lambda x: x.replace(
        'OponentBaseDynamic', 'OBD'), namelist))
    t = ax.table(cellText=A([""]*(len(namelist)**2), len(namelist)), colLabels=namelist,
                 rowLabels=namelist, loc="center", cellColours=cellColours)
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    ax.axis("off")
    plt.show()
    fig, ax = plt.subplots(1, 1)
    ax.bar([x[0] for x in result], [x[1] for x in result], color="r")


if __name__ == '__main__':
    battleList = [Chiang1, Chiang2, Middle, Min, Max,
                  Random, ES, ME, Chen5, Chen6, Bandit, QLearning, OponentBase, OponentBaseDynamic]
    battleList = [QLearning, Bandit, PolicyGradient]
    NNContest(battleList)
    GShow()
