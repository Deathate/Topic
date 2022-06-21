
import sfztest_2agent as model
import types
import numpy as np

model.decayed_step = 10000  # 2500
# 0 -> power decayed, 1 -> double linear decayed, 2 -> linear decayed, 3 -> 1/x-square, 4 ->  1/x
model.EPOCH = 12000
model.ep_mode = 1
model.SHOW_REV_GRAPH = False
model.action_step = 0.1
model.SILENT = True


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
    global minit, m2init
    m = model.sfz_dynamic_model(7706, agentId=0, dynamicGraph=0, verbose=0)
    m2 = model.sfz_dynamic_model(7706, agentId=1, dynamicGraph=0, verbose=0)
    m.ExFactorOnPi = types.MethodType(ExFactorOnPi, m)
    m2.ExFactorOnPi = types.MethodType(ExFactorOnPi, m2)
    m.OtherState = types.MethodType(OtherState, m)
    m2.OtherState = types.MethodType(OtherState, m2)

    g1 = m.Generator()
    g2 = m2.Generator()

    model.N = cid
    minit = model.GetInitialPriceSet()
    m2init = model.GetInitialPriceSet()

    try:
        while True:
            x = next(g1)
            minit = x
            y = next(g2)
            m2init = y
    except StopIteration:
        # print(m.GetFinialChoice(), m2.GetFinialChoice())
        # data = []
        # tbl = dict(m.GetQTable())
        # for x, y in sorted(tbl.items()):
        #     print(x)
        #     y = dict(sorted(y.items(), key=lambda x: x[1], reverse=True))
        #     for s in y.items():
        #         print(f"{s[0]} | {round(s[1], 2)}")
        #     # data.append(list(y.items())[0][0])
        #     print("----------------------------")
        # print(f"Mean: {np.mean(data)}")
        # D = m.GetHistdata(0)
        # for x in D:
        #     print(len(D[x]))
        pass
    return m.GetFinialChoice(), m2.GetFinialChoice()


first = []
second = []
for i in range(1):
    a, b = Test(i)
    print(a, b)
    first.append(a)
    second.append(b)
e1 = model.get_expected_rev(first)
e2 = model.get_expected_rev(second)
print(first)
print(second)
print(e1, e2)
