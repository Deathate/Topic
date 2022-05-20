
import sfztest_1agent_acc as model
import sfztest_2agent_acc as model2
import numpy as np
import collections
import types

model.EPOCH = 1000
model.GET_NEW_DATA = 0
model.SILENT = 1
model.SHOW_REV_GRAPH = 0
itl = 4
model.itl = itl

model2.decayed_step = 50000  # 2500
# 0 -> power decayed, 1 -> double linear decayed, 2 -> linear decayed, 3 -> 1/x-square, 4 ->  1/x
model2.EPOCH = 50000
model2.ep_mode = 1
model2.SHOW_REV_GRAPH = False
model2.action_step = 0.1
model2.SILENT = True


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

    m = model.sfz_dynamic_model(7706, dynamicGraph=0, verbose=0)
    m.AssignHistdata(hist)
    m.SelectionSpace = types.MethodType(SelectionSpace, m)
    g1 = m.Generator()
    for _ in g1:
        pass
    return m.GetCoefficient()


def ExFactorOnPi(self, an):
    a, b = minit, m2init
    if self.GetId() == 0:
        a = an
    else:
        b = an
    if a == b:
        return 0.5
    if self.agentId == 0:
        return a < b
    elif self.agentId == 1:
        return a > b


def OtherState(self):
    return minit, m2init


minit = m2init = 0


def Test(cid):
    model.N = cid
    model.N2 = cid+1
    c0, c1 = Iter()
    print(c0, c1)
    m = model2.sfz_dynamic_model(7706, agentId=0, dynamicGraph=0, verbose=0)
    m2 = model2.sfz_dynamic_model(7706, agentId=1, dynamicGraph=0, verbose=0)
    m.ExFactorOnPi = types.MethodType(ExFactorOnPi, m)
    m2.ExFactorOnPi = types.MethodType(ExFactorOnPi, m2)
    m.OtherState = types.MethodType(OtherState, m)
    m2.OtherState = types.MethodType(OtherState, m2)
    m.SetCurveCoeffient(c0, c1)
    m2.SetCurveCoeffient(c0, c1)
    g1 = m.Generator()
    g2 = m2.Generator()

    model2.N = cid
    global minit, m2init
    minit = model2.GetInitialPriceSet()
    m2init = model2.GetInitialPriceSet()

    try:
        while True:
            x = next(g1)
            minit = x
            y = next(g2)
            m2init = y
    except StopIteration:
        pass
    return m.GetFinialChoice(), m2.GetFinialChoice()


def TenCus():
    first = []
    second = []
    for i in range(10):
        a, b = Test(i)
        print(a, b)
        first.append(a)
        second.append(b)
        e1 = model2.get_expected_rev(first)
        e2 = model2.get_expected_rev(second)
    print(first)
    print(second)
    print(e1, e2)


TenCus()
