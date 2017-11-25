from OpenGL.GL import *
from OpenGL.GLU import *

import pygame
import sys

from motion_render import MotionRender

class MotionViewer:
    WINDOW_TITLE = "CPSC 526 - Final Project: Chrystiano Araujo & Edoardo Dominici"

    def __init__(self, screen_size = (800, 600)):
        self._screen_size = screen_size
        self._events_cb = {}

        self._register_events()
        self._setup_window()
        self._setup_renderers()

    def run(self):
        while True:
            self._handle_events()
            self._draw_scene()

    def shutdown(self, **_):
        # TODO: Any state must be saved here
        sys.exit()

    def _setup_window(self):
      pygame.init()

      flags = pygame.OPENGL | pygame.DOUBLEBUF
      self.screen = pygame.display.set_mode(self._screen_size, flags)

      pygame.display.set_caption(MotionViewer.WINDOW_TITLE)

      # Used to control the framerate
      self.clock = pygame.time.Clock()

    def _setup_renderers(self):
        self._motion_render = MotionRender()

    def _register_events(self):
        self._events_cb[pygame.QUIT]            = self.shutdown
        self._events_cb[pygame.VIDEORESIZE]     = self._resize
        self._events_cb[pygame.MOUSEBUTTONDOWN] = self._handle_mouse_down
        self._events_cb[pygame.MOUSEBUTTONUP]   = self._handle_mouse_up
        self._events_cb[pygame.MOUSEMOTION]     = self._handle_mouse_motion
        self._events_cb[pygame.KEYDOWN]         = self._handle_key_press

    def _handle_events(self):
        for e in pygame.event.get():
            if e.type in self._events_cb:
                self._events_cb[e.type](**e.dict)

    def _resize(self, size, **_):
        glViewport(0, 0, size[0], size[1])

    def _handle_mouse_down(self, pos, button, **_):
        pass

    def _handle_mouse_up(self, pos, button, **_):
        pass

    def _handle_mouse_motion(self, pos, rel, buttons, **_):
        pass

    def _handle_key_press(self, key, mod, **_):
        if key == pygame.K_ESCAPE:
            self.shutdown()

    def _draw_scene(self):
        time_passed = self.clock.tick()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        self._motion_render.draw()
        #floor_render.draw()

        pygame.display.flip()

if __name__ == "__main__":
    viewer = MotionViewer()
    viewer.run()
