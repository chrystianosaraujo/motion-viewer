import sys
import json


from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PyQt5.QtCore import QUrl
from PyQt5.QtCore import *
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtGui import *

import qt_resources

from motion_graph import MotionGraph


class GraphVisualizePage(QWebEnginePage):
    def __init__(self):
        self._is_loaded = False
        self._motion_graph = None
        QWebEnginePage.__init__(self)

        # self.on_node_select_callbacks = []

    @pyqtSlot()
    def on_node_selected(self, id):
        # for listener in self.on_node_select_listeners:
            # listener(id)

        print(f'User selected node {id}')

    @pyqtSlot(str)
    def log(self, string):
        print(string)

    @pyqtSlot()
    def on_channel_ready(self):
        self._is_loaded = True 
        if self._motion_graph is not None:
            self.load_motion_graph(self._motion_graph)
            self._motion_graph = None


    def init(self):
        self.runJavaScript('init()')

    def load_motion_graph(self, motion_graph):
        if not self._is_loaded:
            self._motion_graph = motion_graph
            return

        self.log('Loading Motion Graph')

        nodes = []
        edges = []

        visited = {}
        def build_rec(node):
            if self._id(node) in visited:
                return
            visited[self._id(node)] = True

            nodes.append({ 'id': self._id(node), 'name': node.label, 'weight': str(len(node.iin) + len(node.out)) })

            for out_edge in node.out:
                try:
                    motion_idx = self._motion_graph._motions.index(out_edge.motion) + 1
                except ValueError:
                    motion_idx = 0

                edges.append({ 'id': self._id(out_edge), 'motion_idx': motion_idx, 'label': out_edge.label,'source': self._id(node), 'target': self._id(out_edge.dst), 'length': str(len(out_edge.frames))})
                build_rec(out_edge.dst)

        for node in motion_graph.get_root_nodes():
            build_rec(node)

        js = f"load_motion_graph('{json.dumps(nodes)}', '{json.dumps(edges)}')"
        self.runJavaScript(js)

    def _id(self, obj): 
        return str(id(obj))

    def set_active_edge(self, edge):
        if self._is_loaded:
            self.runJavaScript(f'set_active_edge({self._id(edge)})')


class MotionGraphVisualizerWidget(QWebEngineView):
    def __init__(self, parent=None):
        QWebEngineView.__init__(self, parent)
        self.setAttribute(Qt.WA_TranslucentBackground);
        self.setWindowFlags(Qt.FramelessWindowHint);
        
        self._page = GraphVisualizePage()
        self.setPage(self._page)

        self._channel = QWebChannel(self._page)
        self._channel.registerObject('page', self._page)
        self._page.setWebChannel(self._channel)

        self.loadFinished.connect(self._on_load_finished)
        self.setUrl(QUrl('qrc:/graph_visualizer/test.html'))


    def _on_load_finished(self):
        self._page.init()


    def load_motion_graph(self, motion_graph):
        self._page.load_motion_graph(motion_graph)

    @property
    def visualizer(self):
        return self._page
