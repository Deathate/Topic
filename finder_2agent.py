
import sfztest_dynamic as model
import numpy as np
model.N2 = 1

model.action_step = 0.05
m = model.sfz_dynamic_model(7706, agentId=0)
m2 = model.sfz_dynamic_model(7706, agentId=1)

# model.GetActionSpace()
# print(model.customer)
# print(model.get_rho(0, .96))
# print(model.get_rho(0, 1.04))
# x = m.LoadHistdata()[0]
# for r in x:
# print(f"{r} {np.mean(x[r])}")
x = m2.LoadHistdata()[4]
for r in x:
    print(f"{r} {np.mean(x[r])} {len(x[r])}")

# print(x)
# print(np.mean(x[.86, .9]))
# print(model.get_rho(1, 0.86))
# x = {(1, 1): [0, 0, 0], (1, 2): [0, 1, 0], (2, 1): [0, 0, 5]}
# print([r for r in x.items() if r[0][0] == 1])
