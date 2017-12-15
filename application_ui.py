# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'application.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

import os

WIDTH = 1920
HEIGHT = 1080

class UIMainWindow(object):
    def __init__(self):
        # Any class that implements all on_* methods
        self.callback_handler = None
        self._motions = []

    def setupUi(self, MainWindow):
        self.main_window = MainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(WIDTH, HEIGHT)
        MainWindow.setMinimumSize(QtCore.QSize(800, 600))
       
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.viewer_widget = MotionViewWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewer_widget.sizePolicy().hasHeightForWidth())
        self.viewer_widget.setSizePolicy(sizePolicy)
        self.viewer_widget.setMinimumSize(QtCore.QSize(WIDTH * 0.66, HEIGHT))
        self.viewer_widget.setStyleSheet("")
        self.viewer_widget.setObjectName("viewer_widget")
        
        self.graph_widget = MotionGraphVisualizerWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graph_widget.sizePolicy().hasHeightForWidth())
        self.graph_widget.setSizePolicy(sizePolicy)
        self.graph_widget.setObjectName("graph_widget")

        self.list_widget = QtWidgets.QListWidget(self.centralwidget)

        MainWindow.setCentralWidget(self.centralwidget)

        self.grid_layout = QtWidgets.QGridLayout(self.centralwidget)
        self.grid_layout.addWidget(self.viewer_widget, 0, 0, 2, 1)
        self.grid_layout.addWidget(self.graph_widget, 0, 1, 1, 1)
        self.grid_layout.addWidget(self.list_widget, 1, 1, 1, 1)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setSpacing(0)

        open_file_a = QtWidgets.QAction('Open .BVH', MainWindow)
        open_file_a.setShortcut('Ctrl+O')
        open_file_a.setStatusTip('Import .BVH')
        open_file_a.triggered.connect(self._on_open_file)

        exit_a = QtWidgets.QAction('Exit', MainWindow)
        exit_a.setShortcut('Alt+F4')

        file_menu = MainWindow.menuBar().addMenu('File')
        file_menu.addAction(open_file_a)
        file_menu.addSeparator()
        file_menu.addAction(exit_a)

        generate_motion_graph_a = QtWidgets.QAction('Generate Motion Graph', MainWindow)
        generate_motion_graph_a.setShortcut('Ctrl+G')
        generate_motion_graph_a.triggered.connect(self._on_generate_motion_graph)

        motion_graph_menu = MainWindow.menuBar().addMenu('Motion Graph')
        motion_graph_menu.addAction(generate_motion_graph_a)

        about_a = QtWidgets.QAction('About', MainWindow)
        about_a.triggered.connect(self._on_about)

        help_menu = MainWindow.menuBar().addMenu('Help')
        help_menu.addAction(about_a)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Motion Viewer", "Motion Viewer"))

    def _on_open_file(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_window, "QFileDialog.getOpenFileName()", "","All Files (*);;Biovision Hierarchy (*.bvh)", options=options)
        if path:
            if self.callback_handler:
                motion = self.callback_handler.on_import_motion(path)
                if motion:
                    self._append_motion(path, motion)
                    self._motions.append(motion)

    def _on_generate_motion_graph(self):
        pass

    def _on_about(self):
        pass

    def _append_motion(self, path, motion):
        item = QtWidgets.QListWidgetItem(self.list_widget)
        item_widget = QtWidgets.QWidget(self.list_widget)
        layout = QtWidgets.QHBoxLayout()

        _, filename = os.path.split(path)

        name = QtWidgets.QLabel(filename)
        play_button = QtWidgets.QPushButton('play')
        remove_button = QtWidgets.QPushButton('remove')

        def on_play():
            if self.callback_handler:
                self.callback_handler.on_play_motion(motion)

        def on_remove():
            if self.callback_handler:
                self.callback_handler.on_remove_motion(motion)
                self.list_widget.takeItem(self._motions.index(motion))
                self._motions.remove(motion)

        play_button.clicked.connect(on_play)
        remove_button.clicked.connect(on_remove)

        layout.addWidget(name)
        layout.addWidget(play_button)
        layout.addWidget(remove_button)
        layout.addStretch()
        layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        item_widget.setLayout(layout)
        item.setSizeHint(item_widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, item_widget)

from motion_graph_widget import MotionGraphVisualizerWidget
from motion_viewer_widget import MotionViewWidget
