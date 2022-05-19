import math
import sfztest_1agent_acc as model
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_inv(x):
    return math.log(x/(1-x))


def objective(x, a, b, c):
    return np.poly1d([a, b, c])(x)


X1 = [.77, 0.83, .89, .95, 1.01]
Y1 = [120.22755783294298, 118.2949402771294,
      120.93688587046788, 107.6071054108811, 89.91608609590818]


def Draw(X, Y):
    coeffient, _ = curve_fit(objective, X, Y)
    a, b, c = coeffient
    plt.plot(model.arange(X[0], X[-1]), objective(
        model.arange(X[0], X[-1]), a, b, c), color='b')
    plt.plot(X, Y, "-o", color='y')
    plt.show()


Draw(X1, Y1)
