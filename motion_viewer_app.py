import sys
import glm
import pickle
import os.path

from application_ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets

from skeleton import AnimatedSkeleton
from motion_graph import MotionGraph
from motion_graph_player import MotionGraphPlayer

MOTION_GRAPH_WINDOW_SIZE = 30
GENERATE_GROUNDTRUTH_MOTION_GRAPH = False

class MotionViewerApp:
    def __init__(self):
        self._app = QtWidgets.QApplication(sys.argv)
        self._main_window = QtWidgets.QMainWindow()

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self._main_window)

        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self.on_next_frame)
        self._update_timer.start(1.0 / 120.0)

        self._motion_graph_player = MotionGraphPlayer()
        self._prev_edge = None

    def run(self):
        self._main_window.show()
        self.create_motion_graph(['data/02/02_02.bvh', 'data/02/02_03.bvh'])
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

        self._motion_graph_player.update(1.0 / 120.0)

        edge, frame, transform = self._motion_graph_player.current_motion_data()

        if self._prev_edge is not edge:
            self._ui.graph_widget.visualizer.set_active_edge(edge)
        self._prev_edge = edge

        self._ui.viewer_widget.add_motion(edge.motion, transform)
        self._ui.viewer_widget.set_current_frame(frame)
        self._ui.viewer_widget.updateGL()

if __name__ == "__main__":
    app = MotionViewerApp()
    app.run()
