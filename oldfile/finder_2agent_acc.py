
import sfztest_2agent_acc as model
import numpy as np
model.N = 10
model.action_step = .1
model.th_expected_rev(1)
# model.customer_best_action_plot()
# model.GetActionSpace()
print(model.get_expected_rev(
    [0.3, 1.0, 0.2, 1.2, 0.5, 0.7, 0.9, 0.8, 0.2, 1.7]))
