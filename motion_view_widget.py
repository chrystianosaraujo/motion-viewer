from PyQt5 import QtWidgets, QtCore, QtGui, QtOpenGL
from PyQt5.QtCore import Qt
import OpenGL.GL as GL

# Internal
from motion_render import MotionRender
from skeleton import AnimatedSkeleton
from camera import FirstPersonCamera
from environment_render import EnvironmentRender

class MotionViewWidget(QtOpenGL.QGLWidget):
    CAMERA_CONTROL_KEYS = (Qt.Key_A, Qt.Key_S,
                           Qt.Key_D, Qt.Key_W)

    def __init__(self, parent):
        QtOpenGL.QGLWidget.__init__(self, parent)

        self._frame = 0
        self._camera = FirstPersonCamera(self.width() / self.height())

        self._update_timer = QtCore.QTimer()
        self._update_timer.timeout.connect(self.on_next_frame)
        self._update_timer.start(1.0 / 120.0)
        self._last_mouse_pos = None
        self._camera_enabled = False

        self.setFocusPolicy(Qt.StrongFocus)

    def shutdown(self, **_):
        self._motion_render.clean_up()

    def initializeGL(self):
        self._background_color = [0.0, 0.0, 0.0, 1.0]
        GL.glClearColor(*self._background_color)
        GL.glClearDepth(1.0)

        self._setup_renderers()

    def paintGL(self):
        self._motion_render.set_render_matrices(self._camera.view, self._camera.projection)
        self._environment_render.set_render_matrices(self._camera.view, self._camera.projection)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._environment_render.draw()
        self._motion_render.draw(self._frame)

    def resizeGL(self, w, h):
        pass

    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        if self._is_camera_control_enabled():
            # TODO(Edoardo): call camera controller
            #self._camera.on_mouse_move(-dx, dy)
            pass

    def keyPressEvent(self, event):
        # It is needed since QT launches Press and Release events repeatedly when
        # a key remains pressed.
        if event.isAutoRepeat():
            return

        if event.key() in MotionViewWidget.CAMERA_CONTROL_KEYS:
            self._enable_camera_control(True)

    def keyReleaseEvent(self, event):
        # It is needed since QT launches Press and Release events repeatedly when
        # a key remains pressed.
        if event.isAutoRepeat():
            return

        if event.key() in MotionViewWidget.CAMERA_CONTROL_KEYS:
            self._enable_camera_control(False)

    def on_update(self, dt):
        self._camera.update(dt)

    def on_next_frame(self):
        self._frame += 1

    def _setup_renderers(self):
        skeleton = AnimatedSkeleton()
        skeleton.load_from_file('data/02/02_01.bvh')

        self._motion_render = MotionRender(skeleton)
        self._environment_render = EnvironmentRender()

    def _enable_camera_control(self, value):
        self._camera_enabled = value

        QApp = QtCore.QCoreApplication.instance()
        mouse_state = Qt.BlankCursor if self._camera_enabled \
                                     else Qt.ArrowCursor

        QApp.setOverrideCursor(mouse_state)

    def _is_camera_control_enabled(self):
        return self._camera_enabled
