import ctypes
import glm
import numpy as np
from enum import Enum
import OpenGL.GL as GL

import time
import mesh
from shader import ShaderProgram

from skeleton import AnimatedSkeleton
from skeleton import NodeType as NT
import debugger as dbg


class MotionRender:
    SUPPORTED_NODE_TYPES = [NT.HIP, NT.ABDOMEN, NT.CHEST, NT.NECK, NT.HEAD,
                            NT.RIGHT_COLLAR, NT.RIGHT_SHOULDER, NT.RIGHT_FOREARM, NT.RIGHT_HAND,
                            NT.LEFT_COLLAR, NT.LEFT_SHOULDER, NT.LEFT_FOREARM, NT.LEFT_HAND,
                            NT.RIGHT_BUTTOCK, NT.RIGHT_THIGH, NT.RIGHT_SHIN, NT.RIGHT_FOOT,
                            NT.LEFT_BUTTOCK, NT.LEFT_THIGH, NT.LEFT_SHIN, NT.LEFT_FOOT,
                            NT.LOWER_BACK, NT.SPINE]

    class VertexAttributes(Enum):
        """ Name of each used vertex attribute"""
        Position = {"name": "in_Position", "location": 0}
        Normal = {"name": "in_Normal", "location": 1}

    def __init__(self, skeleton: AnimatedSkeleton):
        self._setup_render_matrices()
        self._setup_ligthing()
        self._create_shaders()
        self._create_mesh_data()
        self._create_gl_objects()

        self._skeleton = skeleton
        self._motion_cache = {}

        # Precompute frames
        for i in range(self._skeleton.frame_count):
            beg = time.time()
            self._skeleton.traverse(i, None)
            print(f'{i} -> {time.time() - beg}')

        self._uniforms = {}

    def set_render_matrices(self, view, project):
        self._view_matrix = view
        self._proj_matrix = project

    def _on_draw_part(self, ntype, name, transform, length, rest_rot):
        # TODO: Add either warning or render with noticeably different color
        if ntype is None:
            return

        # if ntype == NT.HIP:
        #    print(f'Origin: {transform[3]}')

        if ntype not in MotionRender.SUPPORTED_NODE_TYPES:
            return

        scale = glm.scale(glm.mat4(), glm.vec3(4.0, max(length, 1.0), 4.0))
        model = transform * scale

        # PyGLM still does not have binding for inverseTranpose.
        # glm.mat4(glm.mat3) is just a trick to remove the translation part of the normal
        # matrix. It is needed since glm.mat3 is still not fully supported by PyGLM.
        model_view = self._view_matrix * model
        normal_matrix = glm.transpose(glm.inverse(glm.mat4(glm.mat3(model_view))))
        # Updating uniforms

        GL.glUniformMatrix4fv(self._uniforms['model_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(model))
        GL.glUniformMatrix4fv(self._uniforms['normal_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(glm.mat4().value))

        GL.glBindVertexArray(self._vao)

        num_vertices = int(self._vertices.size / 3)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, num_vertices)

    def draw(self, frame):
        frame = frame % self._skeleton.frame_count

        # TODO: remove these strings
        self._uniforms = {
            'model_mat_loc': self._shader_program.uniform_location("modelMatrix"),
            'view_mat_loc': self._shader_program.uniform_location("viewMatrix"),
            'proj_mat_loc': self._shader_program.uniform_location("projectionMatrix"),
            'normal_mat_loc': self._shader_program.uniform_location("normalMatrix"),
            'diffuse_color_loc': self._shader_program.uniform_location("diffuseColor"),
            'ambient_color_loc': self._shader_program.uniform_location("ambientColor"),
            'specular_color_loc': self._shader_program.uniform_location("specularColor"),
            'shininess_loc': self._shader_program.uniform_location("shininess")
        }

        # Setting all shared data
        self._shader_program.bind()
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glUniformMatrix4fv(self._uniforms['view_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(self._view_matrix))
        GL.glUniformMatrix4fv(self._uniforms['proj_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(self._proj_matrix))
        GL.glUniform4fv(self._uniforms['ambient_color_loc'], 1, np.ascontiguousarray(self._ambient_color))
        GL.glUniform4fv(self._uniforms['diffuse_color_loc'], 1, np.ascontiguousarray(self._diffuse_color))
        GL.glUniform4fv(self._uniforms['specular_color_loc'], 1, np.ascontiguousarray(self._specular_color))
        GL.glUniform1f(self._uniforms['shininess_loc'], self._shininess)

        self._skeleton.traverse(frame, self._on_draw_part)

        return

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

        self._view_matrix = glm.mat4()
        self._proj_matrix = glm.mat4()

    def _setup_ligthing(self):
        """Setup shading colors"""

        self._diffuse_color = glm.vec4(0.26, 0.80, 0.26, 1.0)
        self._ambient_color = self._diffuse_color * 0.3
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
        self._normals = np.asarray(normals, dtype=np.float32)

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
