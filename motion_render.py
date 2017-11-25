import OpenGL.GL as GL
import ctypes
import numpy as np
from enum import Enum

from shader import ShaderProgram

class MotionRender:
    class VertexAttributes(Enum):
        """ Name of each used vertex attribute"""

        Position = "in_Position"

    def __init__(self):
        self._create_shaders()
        self._create_mesh_data()
        self._create_gl_objects()

    def draw(self):
        pass

    def _create_shaders(self):
        """Creates the shader program"""

        self._shader_program = ShaderProgram(self._get_vertex_shader_fn(),
                                             self._get_fragent_shader_fn())

        self._shader_program.compile()

    def _create_mesh_data(self):
        """All mesh data is created by this function. So far, it is basically creating a
        cube that will be used to draw the squared characters. Each character's body part
        is drawn as cube.

        TODO: It should be generalized to support multiple rendering styles.
        """

        self.vertex_data = np.array([-0.5, -0.5, -0.5,
                                      0.5, -0.5, -0.5,
                                      0.5, -0.5,  0.5,
                                     -0.5, -0.5,  0.5,
                                     -0.5,  0.5, -0.5,
                                      0.5,  0.5, -0.5,
                                      0.5,  0.5,  0.5,
                                     -0.5,  0.5,  0.5],
                                    dtype=GL.GLfloat)

        self.index_data = np.array([2, 1, 0,   # Bottom Face
                                    3, 2, 0,   # Bottom Face
                                    6, 5, 4,   # Top Face
                                    7, 6, 4,   # Top Face
                                    1, 2, 5,   # Right Face
                                    5, 2, 6,   # Right Face
                                    0, 4, 3,   # Left Face
                                    4, 7, 3,], # Left Face
                                   dtype=GL.GLshort)

    def _create_gl_objects(self):
        """This function creates and initiliazes all opengl objects needed to render the character."""

        self.vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.vao)

        # Generate buffer object for the mesh vertices
        self.vertex_bo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vertex_bo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self.vertex_data.nbytes,
                        self.vertex_data, GL.GL_STATIC_DRAW)

        # Setup Vertex Attrib Pointer
        pos_attrib_name = MotionRender.VertexAttributes.Position.value
        pos_attrib_pointer = self._shader_program.get_attrib_location(pos_attrib_name)

        GL.glEnableVertexAttribArray(pos_attrib_pointer)
        GL.glVertexAttribPointer(pos_attrib_pointer, 3, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))

        # Setup Element Buffer Object
        self.index_bo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.index_bo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, self.index_data.nbytes,
                        self.index_data, GL.GL_STATIC_DRAW)

        # Unbind each used GL object
        GL.glBindVertexArray(0)
        GL.glDisableVertexAttribArray(pos_attrib_pointer)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _get_vertex_shader_fn(self):
        return "./shaders/squared_character.vs"

    def _get_fragent_shader_fn(self):
        return "./shaders/squared_character.fs"
