# External
import OpenGL.GL as GL
import pyglet
import glm

# Python
import sys

# Internal
from motion_render import MotionRender
from skeleton import AnimatedSkeleton
from camera import FirstPersonCamera
from environment_render import EnvironmentRender


class MotionViewer(pyglet.window.Window):
    WINDOW_TITLE = "CPSC 526 - Final Project: Chrystiano Araujo & Edoardo Dominici"

    def __init__(self, screen_size=(1600, 900)):
        self._screen_size = screen_size

        caption = "CPSC 526 - Final Project: Chrystiano Araujo & Edoardo Dominici"
        vsync_enabled = False

        platform = pyglet.window.get_platform()
        display = platform.get_default_display()
        screen = display.get_screens()[0]
        config = screen.get_best_config()
        config.samples = 4
        print(config)

        init_data = {
            "width": screen_size[0],
            "height": screen_size[1],
            "caption": caption,
            "resizable": True,
            "vsync": vsync_enabled,
            "config": config
        }

        super(MotionViewer, self).__init__(**init_data)

        self._fps_display = pyglet.clock.ClockDisplay()

        self._events_cb = {}
        self._background_color = [0.0, 0.0, 0.0, 1.0]

        self._camera = FirstPersonCamera(screen_size[0] / screen_size[1])

        self._frame = 0

        self._setup_gl()
        self._setup_renderers()
        self.set_exclusive_mouse(True)

        pyglet.clock.schedule_interval(self.on_update, 1.0 / 120)
        pyglet.clock.schedule_interval(self.on_next_frame, 1.0 / 120)
        pyglet.clock.set_fps_limit(120)

    def shutdown(self, **_):
        self._motion_render.clean_up()

        # TODO: Any state must be saved here
        sys.exit()

    def on_update(self, dt):
        self._camera.update(dt)

    def on_next_frame(self, dt):
        self._frame += 1

    def on_draw(self):
        pyglet.clock.tick()

        # Update model, view, and project matrices in all renderers
        # TODO: replace this for a camera manipulator
        self._motion_render.set_render_matrices(self._camera.view, self._camera.projection)
        self._environment_render.set_render_matrices(self._camera.view, self._camera.projection)

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self._environment_render.draw()
        self._motion_render.draw(self._frame)
        self._fps_display.draw()

        if self._frame % 10 == 0:
            print(f'{pyglet.clock.get_fps()}')

    def on_mouse_motion(self, x, y, dx, dy):
        self._camera.on_mouse_move(-dx, dy)

    def on_key_press(self, symbol, mod):
        self._camera.on_key_down(symbol)

    def on_key_release(self, symbol, mod):
        if symbol == pyglet.window.key.Q:
            pyglet.app.exit()
        if symbol == pyglet.window.key.R:
            self._frame = 0
        if symbol == pyglet.window.key.N:
            self._frame += 1
        if symbol == pyglet.window.key.M:
            self._frame -= 1

        self._camera.on_key_up(symbol)

    def _setup_gl(self):
        GL.glClearColor(*self._background_color)
        GL.glClearDepth(1.0)

    def _setup_renderers(self):
        skeleton = AnimatedSkeleton()
        skeleton.load_from_file('data/05/05_03.bvh')

        self._motion_render = MotionRender(skeleton)
        self._environment_render = EnvironmentRender()


if __name__ == "__main__":
    viewer = MotionViewer()
    pyglet.app.run()
