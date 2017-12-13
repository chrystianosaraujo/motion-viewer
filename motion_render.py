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
    COLORS = {
        NT.HEAD : glm.vec4(213/255, 0.0, 249.0/255, 1.0),
        NT.EYE : glm.vec4(1.0, 1.0, 1.0, 1.0),
        NT.NECK : glm.vec4(213/255, 0.0, 249.0/255, 1.0),
        NT.TORSO : glm.vec4(1.0, 193.0 / 255, 7.0 / 255, 1.0),
        NT.UPPER_LEG : glm.vec4(63.0/255, 81.0/255, 181.0/255, 1.0),
        NT.LOWER_LEG : glm.vec4(63.0/255, 81.0/255, 181.0/255, 1.0),
        NT.FOOT : glm.vec4(26.0/255, 35.0/255, 126.0/255, 1.0),
        NT.UPPER_ARM : glm.vec4(63.0/255, 81.0/255, 181.0/255, 1.0),
        NT.LOWER_ARM : glm.vec4(63.0/255, 81.0/255, 181.0/255, 1.0),
        NT.FINGER :glm.vec4(26.0/255, 35.0/255, 126.0/255, 1.0),
        NT.HAND : glm.vec4(26.0/255, 35.0/255, 126.0/255, 1.0)
    }

    class VertexAttributes(Enum):
        """ Name of each used vertex attribute"""
        Position = {"name": "in_Position", "location": 0}
        Normal = {"name": "in_Normal", "location": 1}

    def __init__(self, skeleton=None):
        self._setup_render_matrices()
        self._setup_ligthing()
        self._create_shaders()
        self._create_mesh_data()
        self._create_gl_objects()

        self._skeleton = None
        self._skeleton_trans = None
        self._motion_cache = {}

        self.add_motion(skeleton)
        self._uniforms = {}
        self._character_color = None

        self._debug_characters = []
        
    def set_render_matrices(self, view, project):
        self._view_matrix = view
        self._proj_matrix = project

    def _on_draw_part(self, ntype, name, transform, length, rest_rot):
        scale = glm.scale(glm.mat4(), glm.vec3(0.8, max(length, 0.8), 0.8))
        model = transform * rest_rot * scale

        # PyGLM still does not have binding for inverseTranpose.
        # glm.mat4(glm.mat3) is just a trick to remove the translation part of the normal
        # matrix. It is needed since glm.mat3 is still not fully supported by PyGLM.
        model_view = self._view_matrix * model
        normal_matrix = glm.transpose(glm.inverse(glm.mat4(glm.mat3(model_view))))
        # Updating uniforms

        GL.glUniformMatrix4fv(self._uniforms['model_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(model))
        GL.glUniformMatrix4fv(self._uniforms['normal_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(glm.mat4().value))
        
        if ntype:
            GL.glUniform4fv(self._uniforms['ambient_color_loc'], 1, np.ascontiguousarray(self._character_color))

        GL.glBindVertexArray(self._vao)

        num_vertices = int(self._vertices.size / 3)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, num_vertices)

        if ntype:
            GL.glUniform4fv(self._uniforms['ambient_color_loc'], 1, np.ascontiguousarray(self._ambient_color))

    def set_color(self, color):
        self._character_color = color

    def draw_debug(self):
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

       
        color = [
            glm.vec4(1.0, 0.0, 0.0, 1.0),
            glm.vec4(0.0, 1.0, 0.0, 1.0),
            glm.vec4(0.0, 0.0, 1.0, 1.0),
        ]

        if self._debug_characters:            
            for ii, (edge, t) in enumerate(self._debug_characters):
#                root_transform = glm.translate(glm.mat4(), glm.vec3(ii * 10, 0.0, ii * 10))
                self.set_color(color[ii])
                for jj, frame in enumerate(edge.frames):
                    if jj % 5 == 0:
                        edge.motion.traverse(frame, self._on_draw_part, t)


    def draw(self, frame):        
        if self._debug_characters:
            self.draw_debug()

        if self._skeleton is None:
            return

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

        self._skeleton.traverse(frame, self._on_draw_part, self._skeleton_trans)

        return

    def clean_up(self):
        """This function frees all used resources. It includes all VBO's, VAO's, and so on.
        It must be explicitly called since there is no guarantee that there will be a a  OpenGL
        context when this object is garbage collected.
        """

        GL.glDeleteVertexArrays(1, self._vao)
        GL.glDeleteBuffers(1, self._vertex_bo)
        GL.glDeleteBuffers(1, self._normal_bo)

    def add_motion(self, motion, trans = glm.mat4()):
        self._skeleton = motion
        self._skeleton_trans = trans

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
