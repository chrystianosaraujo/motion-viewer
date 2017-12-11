import sys

from application_ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets

class MotionViewer:
    def __init__(self):
        self._app = QtWidgets.QApplication(sys.argv)
        self._main_window = QtWidgets.QMainWindow()

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self._main_window)

    def run(self):
        self._main_window.show()
        sys.exit(self._app.exec_())

if __name__ == "__main__":
    viewer = MotionViewer()
    viewer.run()
