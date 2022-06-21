import sfztest_1agent_acc as model
import tester_1ag_weird as weird
import numpy as np
import pandas as pd

R = []
for x in model.customer:
    R.append((x.c0, x.c1))


def ReduceActionSpace(T, i=0, K=None, F=None):
    if K == None:
        T -= 1
        K = [[x] for x in model.arange(R[i][0], R[i][1])]
        xests, yests = weird.single_company_data()
        F = weird.test(xests, yests, robust=True, graph=False)[-1]
        i += 1
    if i > T:
        return K, F
    selection = model.arange(R[i][0], R[i][1])
    variety = [x+[y] for y in selection for x in K]
    D = dict()
    for x in variety:
        m = round(np.mean([F[xi](xv) for xi, xv in enumerate(x)]), 3)
        if m not in D:
            D[m] = x
        else:
            a, b = np.sum([F[xi](xv) * xv for xi, xv in enumerate(x)]
                          ), np.sum([F[xi](xv) * xv for xi, xv in enumerate(D[m])])
            if a > b:
                D[m] = x
    K = list(D.values())
    return ReduceActionSpace(T, i+1, K, F)


D, F = ReduceActionSpace(10)
S = []
for x in D:
    S.append((round(np.mean(
        [F[xi](xv) for xi, xv in enumerate(x)]), 3), x, model.get_expected_rev(x)))
S = sorted(S, key=lambda x: x[0], reverse=True)
S = pd.DataFrame(S)
S.to_excel("output.xlsx")
