
from collections import namedtuple
import DataCenter as dc
from DataCenter import Starter
import numpy as np
from itertools import combinations, permutations
import time
from multiprocess import Pool
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pandas as pd
import sys
import warnings

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)


def A(arr, v=1):
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    return arr.reshape(-1, v)


def FilterData(**data):
    hist = data["hist"][-1]
    state = namedtuple("d", "mrate, orate, mcopy, ocopy, mrev, orev")
    if data["id"] == 0:
        return state._make((hist.a0, hist.a1, hist.copy0, hist.copy1, hist.rev0, hist.rev1))
    elif data["id"] == 1:
        return state._make((hist.a1, hist.a0, hist.copy1, hist.copy0, hist.rev1, hist.rev0))


def FilterDataAll(**data):
    state = namedtuple("d", "mrate, orate, mcopy, ocopy, mrev, orev")
    d = []
    if data["id"] == 0:
        d = [state._make((hist.a0, hist.a1, hist.copy0, hist.copy1,
                         hist.rev0, hist.rev1)) for hist in data["hist"]]
    elif data["id"] == 1:
        d = [state._make((hist.a1, hist.a0, hist.copy1, hist.copy0,
                         hist.rev1, hist.rev0)) for hist in data["hist"]]
    return d


def Middle(**data):
    return data["currate"] + dc.setting.II / 2


def Max(**data):
    return data["currate"] + dc.setting.II


def Min(**data):
    return data["currate"]


def Random(**data):
    return data["currate"] + rng.uniform(0, 2)


def OponentBase(**data):
    hist = data["hist"]
    if not hist:
        return Random(**data)
    else:
        fd = FilterData(**data)
        return fd.orate + .1


def OponentBaseDynamic(**data):
    hist = data["hist"]

    if not hist:
        return Random(**data)
    else:
        fd = FilterData(**data)
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


def Chen1(**data):
    hist = data["hist"]
    if not hist:
        return Random(**data)
    else:
        if data["id"] == 0:
            myrates = [x.a0 for x in hist]
        elif data["id"] == 1:
            myrates = [x.a1 for x in hist]
        m = np.mean(myrates)
        return m


def Chen2(**data):
    hist = data["hist"]
    if not hist:
        return Random(**data)
    else:
        if data["id"] == 0:
            myrates = [x.a0 for x in hist]
        elif data["id"] == 1:
            myrates = [x.a1 for x in hist]
        m = np.sum(myrates) * 0.5
        return m


def Chen3(**data):
    hist = data["hist"]
    if not hist:
        return Random(**data)
    else:
        hist = hist[-1]
        m = (hist.a0 + hist.a1) / 2
        return m


def ES(**data):
    hist = data["hist"]
    if not hist or len(hist) <= 2:
        return Random(**data)
    else:
        if data["id"] == 0:
            st = np.mean([x.a0 for x in hist[:-1]])
            yt = hist[-1].a0
        elif data["id"] == 1:
            st = np.mean([x.a1 for x in hist[:-1]])
            yt = hist[-1].a1
        alpha = .9
        m = alpha * yt + (1-alpha) * st
        return m


def Chen5(**data):
    hist = data["hist"]
    if not hist:
        return Random(**data)
    else:
        if data["id"] == 0:
            myrates = [x.a1 for x in hist]
        elif data["id"] == 1:
            myrates = [x.a0 for x in hist]
        m = np.mean(myrates) + .2
        return m


def Chen6(**data):
    return Chen3(**data) + .2


def Chiang1(**data):
    hist = data["hist"]
    if not hist or len(hist) <= 10:
        return Random(**data)
    else:
        fd = FilterDataAll(**data)
        X = A([x.orate for x in fd])
        Y = A([x.mrate for x in fd])
        ft = np.mean([x.orate for x in fd]) + .1
        reg = LinearRegression().fit(X, Y)
        m = reg.predict(A(ft))[0, 0]
        return m


def Chiang2(**data):
    hist = data["hist"]
    if not hist or len(hist) <= 10:
        return Random(**data)
    else:
        fd = FilterDataAll(**data)
        X1 = A([x.orate for x in fd])
        X2 = A(data["rates"])
        X = np.hstack((X1, X2))
        Y = A([x.mrate for x in fd])
        ft = [np.mean([x.orate for x in fd[-10:]]) + .1, data["currate"]]
        reg = LinearRegression().fit(X, Y)
        m = reg.predict(A(ft, 2))[0, 0]
        return m


def OneOneContest(ma, mb, show=False):
    rates = dc.rates
    hist = Starter()
    for i in range(1000):
        a0 = ma(currate=rates[i], id=0, hist=hist, rates=rates[:i])
        a1 = mb(currate=rates[i], id=1, hist=hist, rates=rates[:i])
        a0 = np.clip(a0, rates[i], rates[i] + dc.setting.II)
        a1 = np.clip(a1, rates[i], rates[i] + dc.setting.II)
        hist.append(dc.Rev(i, a0, a1))
    pd.DataFrame(hist, columns=[ma.__name__, mb.__name__,
                 'copy0', 'copy1', 'rev0', 'rev1', 'rate']).to_excel("p3/hist.xlsx")
    return dc.Result(hist, "_".join([ma.__name__, mb.__name__]), show=show)


# Middle, Max, Min, Random, OponentBase, Chen1, Chen2, Chen3, Chen4, Chen5, Chen6
battleList = [OponentBaseDynamic, OponentBase, ES,
              Chen1, Chen3, Chen5, Chen6, Chiang1, Chiang2]
# OponentBase, Chen3, OponentBaseDynamic, Chen5
# battleList = [OponentBaseDynamic, Chen3, OponentBase, Chen5]
# battleList = [OponentBase, Chiang1, Chiang2]


def NNContest():
    dc.ClearGraph()
    print("".ljust(40, "-"))
    maxLength = max(
        map(lambda x: len(x), [x.__name__ for x in battleList])) + 2
    comb = list(combinations(battleList, 2))
    results = []
    with Pool() as p:
        for result in tqdm(p.imap(lambda args: OneOneContest(*args), comb), total=len(comb)):
            results.append(result)
    result = {x: 0 for x in battleList}
    for i, x in enumerate(comb):
        contest_result = results[i]
        if contest_result == 1 or contest_result == 3:
            result[x[0]] += 1
        if contest_result == 2 or contest_result == 3:
            result[x[1]] += 1
        # # show process
        txt = [[x[0].__name__, int(contest_result == 1 or contest_result == 3)], [
            x[1].__name__, int(contest_result == 2 or contest_result == 3)]]
        # txt = sorted(txt, key=lambda s: s[0])
        print(
            f"{txt[0][0]:<{maxLength}} vs {txt[1][0]:>{maxLength}}" + f"  --  {txt[0][1]}: {txt[1][1]}")
    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    result = list(map(lambda x: (x[0].__name__, x[1]), result))
    print("".ljust(40, "-"))
    print(result)


if __name__ == '__main__':
    NNContest()
