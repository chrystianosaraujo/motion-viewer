from OpenGL.GL import *
from pyglet.window import key as KEY
from pyglet.window import mouse as MOUSE
import pyglet
import sys
import glm

from motion_render import MotionRender

class MotionViewer(pyglet.window.Window):
    def __init__(self, window_size = (800, 600)):
        self._window_size = window_size

        caption = "CPSC 526 - Final Project: Chrystiano Araujo & Edoardo Dominici"
        vsync_enabled = True

        init_data = {
          "width"    : window_size[0],
          "height"   : window_size[1],
          "caption"  : caption,
          "resizable": True,
          "vsync"    : vsync_enabled,
        }

        super(MotionViewer, self).__init__(**init_data)

        self._fps_display = pyglet.clock.ClockDisplay()

        self._events_cb = {}
        self._background_color = [0.0, 0.0, 0.0, 0.0]

        self._setup_gl()
        self._setup_renderers()

    def shutdown(self, **_):
        self._motion_render.clean_up()

        # TODO: Any state must be saved here
        sys.exit()

    def on_draw(self):
        self.clear()

        # Update model, view, and project matrices in all renderers
        # TODO: replace this for a camera manipulator
        self._motion_render.set_render_matrices(self._view_matrix, self._projection_matrix)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        self._motion_render.draw()
        self._fps_display.draw()

    def _setup_renderers(self):
        self._motion_render = MotionRender()

    def _setup_gl(self):
        glClearColor(*self._background_color)

        aspect = self._window_size[0] / float(self._window_size[1])
        self._projection_matrix = glm.perspective(45.0,
                                                  aspect,
                                                  0.00001,
                                                  100.0)

        self._view_matrix = glm.lookAt(glm.vec3(-1.5, -1.5, -1.0),
                                       glm.vec3( 0.0,  0.0,  0.0),
                                       glm.vec3( 0.0,  1.0,  0.0))

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)

    def on_key_press(self, symbol, modifiers):
        if symbol in (KEY.ESCAPE, KEY.Q):
            self.shutdown()

    def on_mouse_press(self, x, y, button, modifiers):
        """
        See http://pyglet.readthedocs.io/en/pyglet-1.3-maintenance/programming_guide/mouse.html
        """
        pass

    def on_mouse_release(self, x, y, button, modifiers):
        pass

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        pass

if __name__ == '__main__':
    viewer = MotionViewer()
    pyglet.app.run()
