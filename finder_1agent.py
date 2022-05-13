
import sfztest_1agent as model
import numpy as np

model.N2 = 1
m = model.sfz_dynamic_model(7706)


# model.GetActionSpace()
# print(model.customer)
# print(model.get_rho(0, .96))
# print(model.get_rho(0, .99))
x = m.LoadHistdata()[0]
x = dict(sorted(x.items()))

for r in x:
    print(
        f"{r:<5} {len(x[r]):<5} est: {round(np.mean(x[r]),4):<5}  real: {round(model.get_rho(0, r),4)}")
    ns = np.random.binomial(300, model.get_rho(0, r)) / 300
    print(round(ns, 4))
    print("-----------------")
