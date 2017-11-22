from OpenGL.GL import *
from OpenGL.GLU import *

import pygame

class MotionViewer:
    WINDOW_TITLE = "CPSC 526 - Final Project: Chrystiano Araujo & Edoardo Dominici"

    def __init__(self, screen_size = (800, 600)):
        self._screen_size = screen_size
        self._events_cb = {}

        self._setup_window()

    def run(self):
        while not self.quit:
            self._handle_events()
            self._draw_scene()

    def shutdown(self):
        self.quit = True

    def _setup_window(self):
      pygame.init()

      flags = pygame.OPENGL | pygame.DOUBLEBUF
      self.screen = pygame.display.set_mode(self._screen_size, flags)

      pygame.display.set_caption(MotionViewer.WINDOW_TITLE)

      # Used to control the framerate
      self.clock = pygame.time.Clock()

      # TODO: Is there any other way to do this?
      self.quit = False

    def _handle_events(self):
        # TODO: Map types to callbacks
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.shutdown()
            
            elif e.type == pygame.VIDEORESIZE:
                print(e, type(e))
                self._resize(e.size)
            
            elif e.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_down(e.pos, e.button)
            
            elif e.type == pygame.MOUSEBUTTONUP:
                self._handle_mouse_down(e.pos, e.button)
            
            elif e.type == pygame.MOUSEMOTION:
                self._handle_mouse_motion(e.pos, e.rel, e.buttons)

    def _resize(self, size):
        print("RESIZE Event: {0}".format(str(size)))
        glViewport(0, 0, size[0], size[1])

    def _handle_mouse_down(self, pos, button):
        print("MouseDown Event: {0} - {1}".format(str(pos), str(button)))

    def _handle_mouse_up(self, pos, button):
        print("MouseUp Event: {0} - {1}".format(str(pos), str(button)))

    def _handle_mouse_motion(self, pos, rel, buttons):
        print("MouseUp Event: {0} - {1} - {2}".format(str(pos), str(rel), str(buttons)))

    def _draw_scene(self):
        time_passed = self.clock.tick()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        pygame.display.flip()

if __name__ == "__main__":
    viewer = MotionViewer()
    viewer.run()
