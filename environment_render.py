# External
import ctypes
import OpenGL.GL as GL
import numpy as np
from enum import Enum
import glm

# Motion-Viewer
from shader import ShaderProgram
import mesh


class EnvironmentRender:
    class VertexAttributes(Enum):
        """ Name of each used vertex attribute"""
        Position = {"name": "in_Position", "location": 0}
        Normal = {"name": "in_Normal", "location": 1}

    def __init__(self):
        self._shader_program = ShaderProgram(self._get_floor_vs_fn(), self._get_floor_fs_fn())
        for _, attrib in EnvironmentRender.VertexAttributes.__members__.items():
            self._shader_program.bind_attribute(attrib.value["location"], attrib.value["name"])
        self._shader_program.compile()

        vertices, normals = mesh.create_cube_mesh()
        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._normals = np.asarray(normals, dtype=np.float32)

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        # Generate buffer object for the mesh vertices
        self._vertex_bo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vertex_bo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._vertices.nbytes,
                        self._vertices, GL.GL_STATIC_DRAW)

        # Setup Vertex Attrib Pointer
        pos_attrib_pointer = EnvironmentRender.VertexAttributes.Position.value["location"]
        GL.glEnableVertexAttribArray(pos_attrib_pointer)
        GL.glVertexAttribPointer(pos_attrib_pointer, 3, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        self._normal_bo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._normal_bo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._normals.nbytes,
                        self._normals, GL.GL_STATIC_DRAW)

        normal_attrib_pointer = EnvironmentRender.VertexAttributes.Normal.value["location"]
        GL.glEnableVertexAttribArray(normal_attrib_pointer)
        GL.glVertexAttribPointer(normal_attrib_pointer, 3, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        # Unbind each used GL object
        GL.glBindVertexArray(0)
        GL.glDisableVertexAttribArray(pos_attrib_pointer)
        GL.glDisableVertexAttribArray(normal_attrib_pointer)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def set_render_matrices(self, view, projection):
        self._view_matrix = view
        self._proj_matrix = projection

    def draw(self):
        self._shader_program.bind()
        GL.glEnable(GL.GL_DEPTH_TEST)

        model_loc = self._shader_program.uniform_location('modelMatrix')
        view_loc = self._shader_program.uniform_location('viewMatrix')
        proj_loc = self._shader_program.uniform_location('projectionMatrix')

        FLOOR_SIZE = 1000
        scale = glm.scale(glm.mat4(), glm.vec3(FLOOR_SIZE, 1, FLOOR_SIZE))

        GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, np.ascontiguousarray(scale))
        GL.glUniformMatrix4fv(view_loc, 1, GL.GL_FALSE, np.ascontiguousarray(self._view_matrix))
        GL.glUniformMatrix4fv(proj_loc, 1, GL.GL_FALSE, np.ascontiguousarray(self._proj_matrix))

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, int(self._vertices.size / 3))

    def _get_floor_vs_fn(self):
        return './shaders/floor.vs'

    def _get_floor_fs_fn(self):
        return './shaders/floor.fs'
