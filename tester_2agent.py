
import sfztest_dynamic as model
import types


VERSION = model.VERSION
model.EPOCH = 10000
model.decayed_step = 5000  # 2500
# 0 -> power decayed, 1 -> double linear decayed, 2 -> linear decayed, 3 -> 1/x-square, 4 ->  1/x
model.ep_mode = 1
model.CONVCEIL = 100
model.CONVBOUND = 5
model.N = 5
model.CLUSTER = 200
model.SHOW_REV_GRAPH = False

m = model.sfz_dynamic_model(0, agentId=0, dynamicGraph=1, verbose=0)
m2 = model.sfz_dynamic_model(7706, agentId=1, dynamicGraph=0, verbose=0)

minit = model.GetInitialPriceSet()
m2init = model.GetInitialPriceSet()


def ExFactorOnPi(self):
    a, b = minit[self.GetCurrentCid()], m2init[self.GetCurrentCid()]

    if a == b:
        return 0.5
    if self.agentId == 0:
        return a < b
    elif self.agentId == 1:
        return a > b


m.ExFactorOnPi = types.MethodType(ExFactorOnPi, m)
m2.ExFactorOnPi = types.MethodType(ExFactorOnPi, m2)


def OtherState(self):
    return minit[self.GetCurrentCid()], m2init[self.GetCurrentCid()]


# m.OtherState = types.MethodType(OtherState, m)
# m2.OtherState = types.MethodType(OtherState, m2)


def HistoryState(self, x):
    return self.OtherState()


# m.HistoryState = types.MethodType(HistoryState, m)
# m2.HistoryState = types.MethodType(HistoryState, m2)


def ConvergeCondition(self):
    return m.GetConvCounter() > 100 and m2.GetConvCounter() > 100


m.ConvergeCondition = types.MethodType(ConvergeCondition, m)
m2.ConvergeCondition = types.MethodType(ConvergeCondition, m2)

g1 = m.Generator()
g2 = m2.Generator()

i = 0
try:
    while True:
        for i in range(model.N):
            x = next(g1)
            minit[i] = x
        next(g1)
        for i in range(model.N):
            y = next(g2)
            m2init[i] = y
        next(g2)
except StopIteration:
    m.TextResult()
    m2.TextResult()
    m.SavePickle()
    m2.SavePickle()
    pass
input()
