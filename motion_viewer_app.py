import sys
import glm
import pickle
import os.path

from application_ui import UIMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette, QColor

from skeleton import AnimatedSkeleton, UnsupportedError, FormatError
from motion_graph import MotionGraph
from motion_graph_player import MotionGraphPlayer
from motion_player import MotionPlayer

MOTION_GRAPH_WINDOW_SIZE = 30
GENERATE_GROUNDTRUTH_MOTION_GRAPH = False
UPDATE_FREQUENCY = 120
UPDATE_DT = 1.0 / UPDATE_FREQUENCY

class MotionViewerApp:
    def __init__(self):
        self._app = QtWidgets.QApplication(sys.argv)
        self._main_window = QtWidgets.QMainWindow()

        self._ui = UIMainWindow()
        self._ui.setupUi(self._main_window)

        # Hooking up callbacks
        self._ui.callback_handler = self

        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self.on_next_frame)
        self._update_timer.start(UPDATE_DT)

        self._setup_theme()

        self._motion_player = MotionPlayer()
        self._motion_graph_player = MotionGraphPlayer()
        self._prev_edge = None 

    def run(self):
        self._main_window.show()
        self.create_motion_graph(['data/02/02_02.bvh', 'data/02/02_04.bvh'])
        sys.exit(self._app.exec_())

    def load_bvh(self, fn):
        skeleton = AnimatedSkeleton()
        skeleton.load_from_file(fn)
        self._ui.viewer_widget.add_motion(skeleton, glm.mat4())

    def create_motion_graph(self, filenames):
        if os.path.isfile("cache.p"):
            self._motion_graph = pickle.load(open("cache.p", "rb" ))
        else:
            self._motion_graph = MotionGraph(MOTION_GRAPH_WINDOW_SIZE)
            for fn in filenames:
                motion = AnimatedSkeleton()
                motion.load_from_file(fn)
                self._motion_graph.add_motion(motion)

            def progress_cb(factor):
                print("[DEBUG] Building MotionGraph ({:.2f})%".format(factor * 100.0))

            self._motion_graph.build(progress_cb)
            self._motion_graph.serialize()

        if GENERATE_GROUNDTRUTH_MOTION_GRAPH:
            self._generate_groundtruth_data()

        self._ui.graph_widget.load_motion_graph(self._motion_graph)
        self._motion_graph_player.motion_graph = self._motion_graph

    def on_next_frame(self):
        if self._motion_graph is None:
            return

        if self._motion_player.motion is None:
            self._motion_graph_player.update(UPDATE_DT)
            edge, frame, transform = self._motion_graph_player.current_motion_data()
            if self._prev_edge is not edge:
                self._ui.graph_widget.visualizer.set_active_edge(edge)
            self._prev_edge = edge
            
            motion = edge.motion
        else:
            self._motion_player.update(UPDATE_DT)
            # TODO notify visualizer that motion graph is not being played anymore
            motion, frame, transform = self._motion_player.current_motion_data()

        self._ui.viewer_widget.add_motion(motion, transform)
        self._ui.viewer_widget.set_current_frame(frame)
        self._ui.viewer_widget.updateGL()

    # https://gist.github.com/lschmierer/443b8e21ad93e2a2d7eb
    def _setup_theme(self):
        self._app.setStyle("Fusion")
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QtCore.Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QtCore.Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, QtCore.Qt.white)
        dark_palette.setColor(QPalette.Text, QtCore.Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QtCore.Qt.white)
        dark_palette.setColor(QPalette.BrightText, QtCore.Qt.red)
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QtCore.Qt.black)
        self._app.setPalette(dark_palette)
        self._app.setStyleSheet("QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }")

    def on_import_motion(self, path):
        skeleton = AnimatedSkeleton()
        try:
            skeleton.load_from_file(path)
            return skeleton
        except (OSError, UnsupportedError, FormatError) as e:
            self._warn('Import Motion', str(e))

    def on_remove_motion(self, motion):
        if self._motion_player.motion is motion:
            self._motion_player.motion = None

    def on_play_motion(self, motion):
        self._motion_player.motion = motion

    def _warn(self, title, msg):
        QtWidgets.QMessageBox.warning(self._ui.main_window, title, msg)


if __name__ == "__main__":
    app = MotionViewerApp()
    app.run()
