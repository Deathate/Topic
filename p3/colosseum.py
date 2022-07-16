
import DataCenter as dc
from DataCenter import Starter
import numpy as np
from itertools import combinations, permutations
import time

rng = np.random.default_rng(0)


def A(d, pos):
    return np.array(d)[pos]


def Middle(**data):
    return data["currate"] + 1


def Max(**data):
    return data["currate"] + 2


def Min(**data):
    return data["currate"]


def Random(**data):
    return data["currate"] + rng.uniform(0, 2)


def OponentBase(**data):
    return data["oplast"][0] - 0.1


def OneOneContest(ma, mb, show):
    rates = dc.rates
    hist = Starter()
    for i in range(1000):
        lasthist = hist[-1]
        a0 = ma(currate=rates[i], oplast=[
                lasthist.a1, lasthist.copy1, lasthist.rev1])
        a1 = mb(currate=rates[i], oplast=[
                lasthist.a0, lasthist.copy0, lasthist.rev0])
        hist.append(dc.Rev(i, a0, a1))

    return dc.Result(hist, "_".join([ma.__name__, mb.__name__]), show=show)


battleList = []
# battleList = [Middle, Max, Min, Random]
battleList = battleList + [Middle, OponentBase]
battleList = list(set(battleList))

# txt = [["c", 1], ["b", 2]]
# print(sorted(txt, key=lambda x: x[0]))
# exit()


def NNContest():
    result = {x: 0 for x in battleList}
    print("".ljust(40, "-"))
    maxLength = max(
        map(lambda x: len(x), [x.__name__ for x in battleList])) + 2
    for x in combinations(battleList, 2):
        contest_result = OneOneContest(*x, show=len(battleList) == 2)
        if contest_result:
            result[x[0]] += 1
        else:
            result[x[1]] += 1
        # show process
        win = int(contest_result)
        txt = [[x[0].__name__, win], [x[1].__name__], 1 - win]
        print(txt)
        print(sorted(txt, key=lambda s: s[0]))
        print(
            f"{x[0].__name__:<{maxLength}} vs {x[1].__name__:>{maxLength}}" + f"  --  {win}: {1-win}")

    result = sorted(result.items(), key=lambda x: x[1], reverse=True)
    result = list(map(lambda x: (x[0].__name__, x[1]), result))
    print("".ljust(40, "-"))
    print(result)


NNContest()
