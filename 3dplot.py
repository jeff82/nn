# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 09:37:14 2016

@author: ZM
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.figure import Figure
from PyQt4 import  QtCore, QtGui
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from PyQt4.QtGui import QWidget, QPushButton, QMainWindow, QMdiArea, QVBoxLayout, QApplication
from PyQt4.QtCore import Qt

from pylab import *
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

x,y=np.mgrid[-2:2:20j,-2:2:20j]
z=x*np.exp(-x**2-y**2)

ax=plt.subplot(111,projection='3d')
ax.plot_surface(x,y,z,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')










class MyMainWindow(QMainWindow):

    def __init__(self, parent=None):
        """
        """

        super(MyMainWindow,self).__init__(parent)
        self.setWidgets()

    def setWidgets(self, ):


        vBox = QVBoxLayout()
        mainFrame = QWidget()

        self._plotGraphButton = QPushButton("Plot Random Graph")
        self._plotGraphButton.clicked.connect(self.plotRandom)

        self._fig = figure(facecolor="white")
        self._ax = self._fig.add_subplot(111)

        self._canvas = FigureCanvas(self._fig)
        self._canvas.setParent(mainFrame)
        self._canvas.setFocusPolicy(Qt.StrongFocus)


        vBox.addWidget(self._plotGraphButton)
        vBox.addWidget(self._canvas)
#        vBox.addWidget(NavigationToolbar(self._canvas,mainFrame))
        mainFrame.setLayout(vBox)
        self.setCentralWidget(mainFrame)

    def plotRandom(self, ):
        """
        """

        x = linspace(0,4*pi,1000)

        self._ax.plot(x,sin(2*pi*rand()*2*x),lw=2)
        self._canvas.draw()
        
        x,y=np.mgrid[-2:2:20j,-2:2:20j]
        z=x*np.exp(-x**2-y**2)
        
        self._ax.plot.subplot(111,projection='3d')
        self.plot_surface(x,y,z,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')


if __name__ == '__main__':
    qApp = QApplication(sys.argv)
    MainWindow = MyMainWindow()

    MainWindow.show()
    sys.exit(qApp.exec_())