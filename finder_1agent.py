
import sfztest_dynamic as model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom

model.N = 1
m = model.sfz_dynamic_model(7706, agentId=0)


# model.GetActionSpace()
# print(model.customer)
# print(model.get_rho(0, .96))
# print(model.get_rho(0, .99))
x = m.LoadHistdata()[0]
for r in x:
    print(
        f"{r:<5} {len(x[r]):<5} {round(np.mean(x[r]),4):<5}  {round(model.get_rho(0, r),4)}")
    ns = np.random.binomial(len(x[r]), model.get_rho(0, r)) / len(x[r])
    print(round(ns, 4))
    print("---")

for i in range(10):
    ns = np.random.binomial(len(x[r]), model.get_rho(0, r)) / len(x[r])
    print(ns)
# print(x)
# print(np.mean(x[.86, .9]))
# print(model.get_rho(1, 0.86))
# x = {(1, 1): [0, 0, 0], (1, 2): [0, 1, 0], (2, 1): [0, 0, 5]}
# print([r for r in x.items() if r[0][0] == 1])

# sns.distplot(np.random.binomial(
#     n=1971, p=0.1279, size=1000)/1971, hist=True, kde=False)
# sns.distplot(binom.rvs(1971, .1279, size=1000)/1971, hist=True, kde=False)
# plt.show()
