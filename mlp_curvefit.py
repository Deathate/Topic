import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.losses import Loss
from keras import Sequential
from keras.layers import Activation, Dense, Dropout
import matplotlib.pyplot as plt
from rho_simulate_weird import CreateEstimate, xset, yset
from sfztest_1agent_acc import arange
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def my_huber_loss(y_true, y_pred):
    error = y_true - y_pred
    x = tf.abs(error)
    return tf.math.reduce_variance(x)


setid = 1
X = xset[setid]
Y = yset[setid]
itl = 0.05
size = len(arange(min(X), max(X), itl)) * 10
x3est, y3est, spl = CreateEstimate(X, Y, size, itl, 0)
x3est = x3est.repeat(50)
y3est = y3est.repeat(50)

model = Sequential()
model.add(Dense(units=64, input_dim=1, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=8, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(units=1, activation='linear'))

model.compile(loss=my_huber_loss,
              optimizer='adam')

callback = keras.callbacks.EarlyStopping(monitor="loss", patience=50)
start = time.time()
model.fit(x3est, y3est, epochs=200, verbose=0,
          batch_size=10, callbacks=[])
end = time.time()
print(end-start)
# model.summary()

fr = arange(min(x3est), max(x3est), .01)
classes = model.predict(fr, batch_size=1)
plt.plot(fr, classes, c='b')
plt.plot(fr, spl(fr), c='y')
plt.scatter(x3est, y3est, c="r")
plt.show()
