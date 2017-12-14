# External
import ctypes
import OpenGL.GL as GL
from OpenGL.GL.EXT.texture_filter_anisotropic import *
import numpy as np
from enum import Enum
import glm
import PIL

# Motion-Viewer
from shader import ShaderProgram
import mesh


class EnvironmentRender:
    class VertexAttributes(Enum):
        """ Name of each used vertex attribute"""
        Position = {"name": "in_Position", "location": 0}
        Normal = {"name": "in_Normal", "location": 1}
        TexCoord = {"name": "in_TexCoord", "location": 2}

    def __init__(self):
        self._shader_program = ShaderProgram(self._get_floor_vs_fn(), self._get_floor_fs_fn())
        for _, attrib in EnvironmentRender.VertexAttributes.__members__.items():
            self._shader_program.bind_attribute(attrib.value["location"], attrib.value["name"])
        self._shader_program.compile()

        vertices, normals, texcoords = mesh.create_grid_mesh(300, 300, 50)
        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._normals = np.asarray(normals, dtype=np.float32)
        self._texcoords = np.asarray(texcoords, dtype=np.float32)

        print(self._texcoords)

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

        self._texcoord_bo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._texcoord_bo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._texcoords.nbytes, self._texcoords, GL.GL_STATIC_DRAW)

        texcoord_attrib_pointer = EnvironmentRender.VertexAttributes.TexCoord.value["location"]
        GL.glEnableVertexAttribArray(texcoord_attrib_pointer)
        GL.glVertexAttribPointer(texcoord_attrib_pointer, 2, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        # Unbind each used GL object
        GL.glBindVertexArray(0)
        GL.glDisableVertexAttribArray(pos_attrib_pointer)
        GL.glDisableVertexAttribArray(normal_attrib_pointer)
        GL.glDisableVertexAttribArray(texcoord_attrib_pointer)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        floor_image = PIL.Image.open('textures/floor.png')
        floor_texture_data = np.array(list(floor_image.getdata()), np.uint8)
        self.floor_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.floor_texture)
        GL.glTexParameterf(GL.GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0); 
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, floor_image.size[0], floor_image.size[1], 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, floor_texture_data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    def set_render_matrices(self, view, projection):
        self._view_matrix = view
        self._proj_matrix = projection

    def draw(self):
        self._shader_program.bind()
        GL.glEnable(GL.GL_DEPTH_TEST)

        model_loc = self._shader_program.uniform_location('modelMatrix')
        view_loc = self._shader_program.uniform_location('viewMatrix')
        proj_loc = self._shader_program.uniform_location('projectionMatrix')
        sampler_loc = self._shader_program.uniform_location('color_map')

        FLOOR_SIZE = 2
        scale = glm.scale(glm.mat4(), glm.vec3(FLOOR_SIZE, 1, FLOOR_SIZE)) * glm.translate(glm.mat4(), glm.vec3(0.0, -20.0, 0.0))

        GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, np.ascontiguousarray(glm.mat4()))
        GL.glUniformMatrix4fv(view_loc, 1, GL.GL_FALSE, np.ascontiguousarray(self._view_matrix))
        GL.glUniformMatrix4fv(proj_loc, 1, GL.GL_FALSE, np.ascontiguousarray(self._proj_matrix))

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.floor_texture)
        GL.glUniform1i(sampler_loc, 0)

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, int(self._vertices.size / 3))

    def _get_floor_vs_fn(self):
        return './shaders/floor.vs'

    def _get_floor_fs_fn(self):
        return './shaders/floor.fs'
