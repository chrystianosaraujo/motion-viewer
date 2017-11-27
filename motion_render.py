import ctypes
import glm
import numpy as np
from enum import Enum
import OpenGL.GL as GL

import mesh
from shader import ShaderProgram

class MotionRender:
    class VertexAttributes(Enum):
        """ Name of each used vertex attribute"""

        Position = {"name": "in_Position", "location": 0}
        Normal   = {"name": "in_Normal"  , "location": 1}

    def __init__(self):
        self._setup_render_matrices()
        self._setup_ligthing()
        self._create_shaders()
        self._create_mesh_data()
        self._create_gl_objects()

    def set_render_matrices(self, view, project):
        self._view_matrix = view
        self._proj_matrix = project

    def draw(self):
        model_view = self._view_matrix * self._model_matrix

        # PyGLM still does not have binding for inverseTranpose.
        # glm.mat4(glm.mat3) is just a trick to remove the translation part of the normal
        # matrix. It is needed since glm.mat3 is still not fully supported by PyGLM.
        normal_matrix = glm.transpose(glm.inverse(glm.mat4(glm.mat3(model_view))))

        GL.glEnable(GL.GL_DEPTH_TEST)
        self._shader_program.bind()

        # Update uniforms
        # TODO: remove these strings
        model_mat_loc  = self._shader_program.uniform_location("modelMatrix")
        view_mat_loc   = self._shader_program.uniform_location("viewMatrix")
        proj_mat_loc   = self._shader_program.uniform_location("projectionMatrix")
        normal_mat_loc = self._shader_program.uniform_location("normalMatrix")

        diffuse_color_loc  = self._shader_program.uniform_location("diffuseColor")
        ambient_color_loc  = self._shader_program.uniform_location("ambientColor")
        specular_color_loc = self._shader_program.uniform_location("specularColor")
        shininess_loc      = self._shader_program.uniform_location("shininess")

        GL.glUniformMatrix4fv(model_mat_loc , 1, GL.GL_FALSE, glm.value_ptr(self._model_matrix))
        GL.glUniformMatrix4fv(view_mat_loc  , 1, GL.GL_FALSE, glm.value_ptr(self._view_matrix))
        GL.glUniformMatrix4fv(proj_mat_loc  , 1, GL.GL_FALSE, glm.value_ptr(self._proj_matrix))
        GL.glUniformMatrix4fv(normal_mat_loc, 1, GL.GL_FALSE, glm.value_ptr(normal_matrix))

        GL.glUniform4fv(ambient_color_loc  , 1, glm.value_ptr(self._ambient_color))
        GL.glUniform4fv(diffuse_color_loc  , 1, glm.value_ptr(self._diffuse_color))
        GL.glUniform4fv(specular_color_loc , 1, glm.value_ptr(self._specular_color))
        GL.glUniform1f (shininess_loc, self._shininess)

        GL.glBindVertexArray(self._vao)

        num_vertices = int(self._vertices.size / 3)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, num_vertices)

        GL.glBindVertexArray(0)
        self._shader_program.unbind()

    def clean_up(self):
        """This function frees all used resources. It includes all VBO's, VAO's, and so on.
        It must be explicitly called since there is no guarantee that there will be a a  OpenGL
        context when this object is garbage collected.
        """

        GL.glDeleteVertexArrays(1, self._vao)
        GL.glDeleteBuffers(1, self._vertex_bo)
        GL.glDeleteBuffers(1, self._normal_bo)

    def _setup_render_matrices(self):
        """It simply initializes model, view, and projection matrices"""

        self._model_matrix = glm.mat4()
        self._view_matrix  = glm.mat4()
        self._proj_matrix  = glm.mat4()

    def _setup_ligthing(self):
        """Setup shading colors"""

        self._diffuse_color  = glm.vec4(0.26, 0.80, 0.26, 1.0)
        self._ambient_color  = self._diffuse_color * 0.3
        self._specular_color = glm.vec4(0.84, 0.30, 0.74, 1.0)
        self._shininess = 64

    def _create_shaders(self):
        """Creates the shader program"""

        self._shader_program = ShaderProgram(self._get_vertex_shader_fn(),
                                             self._get_fragent_shader_fn())


        # Setup Vertex Attributes
        for _, attrib in MotionRender.VertexAttributes.__members__.items():
            self._shader_program.bind_attribute(attrib.value["location"], attrib.value["name"])

        self._shader_program.compile()

    def _create_mesh_data(self):
        """All mesh data is created by this function. So far, it basically creates one
        cube that will be used to draw the squared characters. Each character's body part
        is drawn as cube.

        TODO: It should be generalized to support multiple rendering styles.
        """
        vertices, normals = mesh.create_cube_mesh()

        self._vertices = np.asarray(vertices, dtype=np.float32)
        self._normals  = np.asarray(normals, dtype=np.float32)

    def _create_gl_objects(self):
        """This function creates and initiliazes all opengl objects needed to render the character."""

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        # Generate buffer object for the mesh vertices
        self._vertex_bo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vertex_bo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._vertices.nbytes,
                        self._vertices, GL.GL_STATIC_DRAW)

        # Setup Vertex Attrib Pointer
        pos_attrib_pointer = MotionRender.VertexAttributes.Position.value["location"]
        GL.glEnableVertexAttribArray(pos_attrib_pointer)
        GL.glVertexAttribPointer(pos_attrib_pointer, 3, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        self._normal_bo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._normal_bo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._normals.nbytes,
                        self._normals, GL.GL_STATIC_DRAW)

        normal_attrib_pointer = MotionRender.VertexAttributes.Normal.value["location"]
        GL.glEnableVertexAttribArray(normal_attrib_pointer)
        GL.glVertexAttribPointer(normal_attrib_pointer, 3, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        # Unbind each used GL object
        GL.glBindVertexArray(0)
        GL.glDisableVertexAttribArray(pos_attrib_pointer)
        GL.glDisableVertexAttribArray(normal_attrib_pointer)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def _get_vertex_shader_fn(self):
        return "./shaders/squared_character.vs"

    def _get_fragent_shader_fn(self):
        return "./shaders/squared_character.fs"
