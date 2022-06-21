import math
import sfztest_1agent_acc as model
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_inv(x):
    return math.log(x/(1-x))


def objective(x, a, b):
    return sigmoid(np.poly1d([a, b])(x))


X1 = [0.79, .85, .9, 1, 1.1, 1.29]
Y1 = [0.9, .85, .85, 0.65, 0.4, 0.1]

X2 = [0.79, .85, .9, 1, 1.05, 1.15, 1.2, 1.29]
Y2 = [0.9, .85, .85, 0.5, 0.2, .2, .2, 0.2]

X3 = [0.79, .85, .9, 1, 1.05, 1.15, 1.2, 1.29]
Y3 = [0.75, .65, .5, 0.3, 0.1, 0, 0, 0]


def Draw(X, Y):
    coeffient, _ = curve_fit(objective, X, Y)
    a, b = coeffient
    print(a, b)
    plt.plot(model.arange(.8, 1.29), objective(
        model.arange(.8, 1.29), a, b), color='b')
    plt.plot(X, Y, "-o", color='y')
    # plt.show()


Draw(X1, Y1)
Draw(X2, Y2)
Draw(X3, Y3)
