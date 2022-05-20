
import sfztest_1agent_acc as model
import numpy as np
import collections
import types

model.EPOCH = 1000
model.GET_NEW_DATA = 0
model.SILENT = 1
model.SHOW_REV_GRAPH = 0

itl = 4
model.itl = itl


def SelectionSpace(self, cid):
    return model.arange(
        model.customer[cid].c0, model.customer[cid].c1, count=itl)


def Iter(seed=0):
    np.random.seed(seed)
    hist = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for i in range(model.N, model.N2):
        xline = model.arange(
            model.customer[i].c0, model.customer[i].c1, count=itl)
        size = 45
        for x in xline:
            hist[i][x] = [np.random.binomial(
                1, model.get_rho(i, x)) for _ in range(size)]
    # fd = [model.get_rho(first, r) for r in xline]
    # cusline = model.arange(model.customer[0].c0, model.customer[0].c1)
    # d = [model.get_rho(0, r) for r in cusline]
    # plt.plot(cusline, d, '--', color='red')
    # plt.plot(cusline, model.Objective(cusline, c0, c1), '-o', color='b')
    # plt.show()

    m = model.sfz_dynamic_model(7706, dynamicGraph=0, verbose=0)
    m.AssignHistdata(hist)
    m.SelectionSpace = types.MethodType(SelectionSpace, m)
    g1 = m.Generator()
    for _ in g1:
        pass
    return m.GetFinalChoice()[0], m.GetFinalRound()


def Test(seed=0):
    c = []
    round = 0
    first = 0
    second = 10
    for i in range(first, second):
        model.N = i
        model.N2 = model.N+1
        a0, a1 = Iter(seed)
        c.append(a0)
        round += a1
    model.N = first
    model.N2 = second
    print(c)
    model.th_expected_rev(1)
    x = model.get_expected_rev(c)
    print(f"Total Round: {round/(model.N2-model.N)}, rev: {x}")
    return round, x


def MeanResult():
    r = 0
    s = 0
    for i in range(10):
        a, b = Test(i)
        r += a
        s += b
    print(s/10, r/100)


Test()
# MeanResult()
