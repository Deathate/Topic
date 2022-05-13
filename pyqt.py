
import pyqtgraph as plt
from pyqtgraph.Qt import QtCore
import random
import time
import math
import threading
import time
import sys
# https://ntpuccw.blog/pyqtgraph/

plt.setConfigOption('background', 'w')
plt.setConfigOption('foreground', 'k')


class DynamicPlotter():

    def __init__(self):
        self.x = [1]
        self.y = [1]

        self.app = plt.mkQApp()
        self.plt = plt.plot(title='Dynamic Plotting with PyQtGraph')
        # self.plt.resize(100, 100)
        self.plt.showGrid(x=True, y=True)
        self.plt.setLabel('left', 'revenue')
        self.plt.setLabel('bottom', 'time', 'round')
        self.curve = self.plt.plot(self.x, self.y, pen=(255, 0, 0))
        # QTimer
        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(lambda: print(random.random()))
        # self.timer.start(500)

    def getdata(self):
        frequency = 0.5
        noise = random.normalvariate(0., 1.)
        new = 10.*math.sin(time.time()*frequency*2*math.pi) + noise
        return new

    def updateplot(self, a, b):
        self.x.append(a)
        self.y.append(b)
        self.curve.setData(self.x, self.y)
        self.app.processEvents()

    def addVerticalLine(self, v, color, width):
        a = plt.InfiniteLine(angle=0, pen=plt.mkPen(color, width=width))
        a.setValue(v)
        self.plt.addItem(a)

    def run(self):
        sys.exit(plt.exec())


def Create():
    plotter = DynamicPlotter()
    threading.Thread(target=lambda: plotter.run()).start()
    return plotter


# Create()
# while(1):
#     input()
