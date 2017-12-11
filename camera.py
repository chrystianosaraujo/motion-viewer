import glm
import math
from PyQt5.QtCore import Qt

class FirstPersonCamera:
    def __init__(self, aspect_ratio):
        self._position = glm.vec3(80, 60, 0.0)
        self._direction = glm.vec3(-1.0, -1.0, -1.0)
        self._velocity = glm.vec3(0.0, 0.0, 0.0)
        self._projection = glm.perspective(45.0, aspect_ratio, 0.01, 1000.0)
        self._speed = 5  # units per second
        self._mouse_sensibility = 0.1

    def on_key_down(self, key):
        if key == Qt.Key_W:
            self._velocity.x += 1
        if key == Qt.Key_S:
            self._velocity.x -= 1
        if key == Qt.Key_A:
            self._velocity.y -= 1
        if key == Qt.Key_D:
            self._velocity.y += 1
        if key == Qt.Key_Space:
            self._velocity.z += 1
        # if key == Qt.Key_LSHIFT:
        #     self._velocity.z -= 1

    def on_key_up(self, key):
        if key == Qt.Key_W:
            self._velocity.x -= 1
        if key == Qt.Key_S:
            self._velocity.x += 1
        if key == Qt.Key_A:
            self._velocity.y += 1
        if key == Qt.Key_D:
            self._velocity.y -= 1
        if key == Qt.Key_Space:
            self._velocity.z -= 1
        # if key == Qt.Key_LSHIFT:
        #     self._velocity.z += 1

    def on_mouse_move(self, dx, dy):
        yaw_axis = glm.vec3(0.0, 1.0, 0.0)
        pitch_axis = glm.cross(self._direction, yaw_axis)

        # TODO radians conversion can be avoided w/ lower sensibility
        yaw_angle = math.radians(-dx * self._mouse_sensibility)
        pitch_angle = math.radians(-dy * self._mouse_sensibility)

        self._direction = (glm.rotate(glm.mat4(1.0), yaw_angle, yaw_axis) * glm.vec4(self._direction, 1.0)).xyz
        self._direction = (glm.rotate(glm.mat4(1.0), pitch_angle, pitch_axis) * glm.vec4(self._direction, 1.0)).xyz

    def on_resize(self, aspect_ratio):
        self._projection = glm.perspective(45.0, aspect_ratio, 0.00001, 5000.0)

    def update(self, ms):  # Add
        up = glm.vec3(0.0, 1.0, 0.0)
        right = glm.cross(self._direction, up)

        self._position += self._direction * self._velocity.x * self._speed
        self._position += right * self._velocity.y * self._speed
        self._position += up * self._velocity.z * self._speed

    @property
    def view(self):
        # Could be cached..
        return glm.lookAt(self._position, self._position + self._direction, glm.vec3(0.0, 1.0, 0.0))

    @property
    def projection(self):
        return self._projection
