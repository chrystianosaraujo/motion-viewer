# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'application.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1848, 1165)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
        MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.viewer_widget = MotionViewWidget(self.centralwidget)
        self.viewer_widget.setGeometry(QtCore.QRect(0, 0, 1920, 1080))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewer_widget.sizePolicy().hasHeightForWidth())
        self.viewer_widget.setSizePolicy(sizePolicy)
        self.viewer_widget.setMinimumSize(QtCore.QSize(1500, 0))
        self.viewer_widget.setStyleSheet("")
        self.viewer_widget.setObjectName("viewer_widget")
        self.graph_widget = MotionGraphVisualizerWidget(self.viewer_widget)
        self.graph_widget.setGeometry(QtCore.QRect(0, 0, 800, 600))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graph_widget.sizePolicy().hasHeightForWidth())
        self.graph_widget.setSizePolicy(sizePolicy)
        self.graph_widget.setMinimumSize(QtCore.QSize(0, 0))
        self.graph_widget.setObjectName("graph_widget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1848, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

from motion_graph_widget import MotionGraphVisualizerWidget
from motion_viewer_widget import MotionViewWidget
