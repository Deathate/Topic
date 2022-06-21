
import sfztest_1agent_acc as model
import numpy as np
import collections

model.EPOCH = 1000
model.decayed_step = 1500  # 1500
model.ep_mode = 1
model.CONVCEIL = 200
model.CONVBOUND = 0
model.CLUSTER = 20
model.GET_NEW_DATA = 1
model.SILENT = 1
model.SHOW_REV_GRAPH = False

itl = 7
model.itl = itl


def Iter():
    np.random.seed(0)
    hist = collections.defaultdict(
        lambda: collections.defaultdict(list))
    xline = model.arange(0, 2, count=itl)
    size = 45
    for i in range(model.N, model.N2):
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
    g1 = m.Generator()
    for _ in g1:
        pass
    # m.TextResult()
    return m.GetFinalChoice()[0], m.GetFinalRound()


c = []
round = 0
first = 0
second = 10
for i in range(first, second):
    model.N = i
    model.N2 = model.N+1
    a0, a1 = Iter()
    c.append(a0)
    round += a1
model.N = first
model.N2 = second
print(c)
model.th_expected_rev(1)
x = model.get_expected_rev(c)
print(f"Total Round: {round/(model.N2-model.N)}, rev: {x}")
