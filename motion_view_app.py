import sys
import glm
import pickle
import os.path

from application_ui import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets

from skeleton import AnimatedSkeleton
from motion_graph import MotionGraph

MOTION_GRAPH_WINDOW_SIZE = 30

class MotionViewerApp:
    def __init__(self):
        self._app = QtWidgets.QApplication(sys.argv)
        self._main_window = QtWidgets.QMainWindow()

        self._ui = Ui_MainWindow()
        self._ui.setupUi(self._main_window)

        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self.on_next_frame)
        self._current_edge = None
        self._current_edge_trans = None
        self._current_frame = 0

        self._beg_edge = None
        self._edge_counter = 0

    def run(self):
        self._main_window.show()

        #self.load_bvh('data/02/02_01.bvh')
        self.create_motion_graph(['data/02/02_02.bvh', 'data/02/02_03.bvh'])
        self._update_timer.start(1.0 / 120.0)

        try:
            self._app.exec_()
        except Exception as e:
            print(str(e))
        finally:
            sys.exit()

    def load_bvh(self, fn):
        skeleton = AnimatedSkeleton()
        skeleton.load_from_file(fn)
        self._ui.scene_widget.add_motion(skeleton, glm.mat4())

    def create_motion_graph(self, filenames):
        if os.path.isfile("cache.p"):
            self._motion_graph = pickle.load(open("cache.p", "rb" ))
        else:
            self._motion_graph = MotionGraph(MOTION_GRAPH_WINDOW_SIZE)
            for fn in filenames:
                motion = AnimatedSkeleton()
                motion.load_from_file(fn)
                self._motion_graph.add_motion(motion)
            self._motion_graph.build()
            self._motion_graph.serialize()

        #mo1, t1 = self._motion_graph.begin_edge(1)
        #self._ui.scene_widget._motion_render._debug_characters.append((mo1, t1))
        #print(t1)
        #mo2, t2 = self._motion_graph.next_edge(mo1)
        #self._ui.scene_widget._motion_render._debug_characters.append((mo2, t2))
        #print(t2)
        #mo3, t3 = self._motion_graph.next_edge(mo2)
        #self._ui.scene_widget._motion_render._debug_characters.append((mo3, t3))
        #print(t3)

    def on_next_frame(self):
        #self._ui.scene_widget.updateGL()        
        #return

        global BREAK
        if self._motion_graph is None:
            return

        if self._current_edge is None:
            self._current_edge, self._current_edge_trans = self._motion_graph.begin_edge(0)
            self._beg_edge = self._current_edge
            print(f'{self._edge_counter} : {self._current_edge.frames}')

        color = [
            glm.vec4(1.0, 0.0, 0.0, 1.0),
            glm.vec4(0.0, 1.0, 0.0, 1.0),
            glm.vec4(0.0, 0.0, 1.0, 1.0),
            glm.vec4(0.0, 1.0, 1.0, 1.0),
            glm.vec4(1.0, 0.0, 1.0, 1.0),
        ]

        path = [0, 0, 0, 0, 1, 0, 0, 0]#, 0, 0, 0, 0]

        #self._ui.scene_widget._motion_render.set_color(color[1])

        self._ui.scene_widget.add_motion(self._current_edge.motion, self._current_edge_trans)
        self._ui.scene_widget.set_current_frame(self._current_edge.frames[self._current_frame])
        self._ui.scene_widget.updateGL()

        print("Frame", self._current_frame, "CurrentEdge", self._edge_counter)
        self._current_frame += 1

        if not self._current_edge.is_valid_frame(self._current_frame):
            if self._edge_counter >= len(path) - 1:
                self._current_edge, self._current_edge_trans = self._motion_graph.begin_edge(0)
                self._edge_counter = 0
                self._current_frame = 0
                print(f'{self._edge_counter} : {self._current_edge.frames}')
            else:
                try:
                    self._current_edge, self._current_edge_trans = self._motion_graph.next_edge(self._current_edge, path[self._edge_counter], self._current_edge_trans)
                except IndexError:
                    pass
                self._current_frame = 0
                self._edge_counter += 1
                print(f'{self._edge_counter} : {self._current_edge.frames}')

if __name__ == "__main__":
    app = MotionViewerApp()
    app.run()