import ctypes
import glm
import numpy as np
from enum import Enum
import OpenGL.GL as GL

import time
import mesh
from shader import ShaderProgram
import motion_render_data

from skeleton import AnimatedSkeleton
from skeleton import NodeType as NT
import debugger as dbg

RENDER_DATA = motion_render_data.get_squared_character_render_data()

class MotionRender:
    class VertexAttributes(Enum):
        """ Name of each used vertex attribute"""
        Position = {"name": "in_Position", "location": 0}
        Normal   = {"name": "in_Normal"  , "location": 1}

    def __init__(self, skeleton=None):
        self._cylinder_mem_info = None
        self._sphere_mem_info = None
        self._skeleton = None
        self._skeleton_trans = None
        self._motion_cache = {}
        self._character_color = glm.vec4(124.0 / 255.0, 174.0 / 255.0,
                                         255.0 / 255.0, 1.0)
        self._joints_color = glm.vec4(200.0 / 255.0, 200.0 / 255.0,
                                      200.0 / 255.0, 1.0)
        self._uniforms = {}

        self._setup_render_matrices()
        self._setup_ligthing()
        self._create_shaders()
        self._create_mesh_data()
        self._create_gl_objects()

        self.add_motion(skeleton)

    def set_render_matrices(self, view, project):
        self._view_matrix = view
        self._proj_matrix = project

    def _on_draw_part(self, ntype, name, transform, length, rest_rot):
        render_data = RENDER_DATA.get(ntype)
        if render_data is None or not render_data.enabled:
            return

        GL.glBindVertexArray(self._vao)

        # Draw body Parts
        scale = glm.scale(glm.mat4(), glm.vec3(0.5, max(length, 0.5), 0.5))
        model = transform * rest_rot * scale

        GL.glUniformMatrix4fv(self._uniforms['model_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(model))
        GL.glUniform4fv(self._uniforms['character_color_loc'], 1, np.ascontiguousarray(self._character_color))

        GL.glDrawArrays(GL.GL_TRIANGLES, self._cylinder_mem_info[0], self._cylinder_mem_info[1])

        # Draw joints
        # Ignoring the following parts
        
        # NodeType.UPPER_ARM: RightShoulder
        # NodeType.UPPER_ARM: LeftShoulder
        # NodeType.TORSO    : RHipJoint
        # NodeType.TORSO    : LHipJoint
        # NodeType.FINGER   : LThumb
        # NodeType.FINGER   : LeftFingerBase
        # NodeType.FINGER   : LeftHandIndex1
        IGNORED_PARTS = [NT.TORSO, NT.UPPER_ARM, NT.FINGER]
        if ntype not in IGNORED_PARTS:
            scale = glm.scale(glm.mat4(), glm.vec3(0.55, 0.55, 0.55))
            model = transform * rest_rot * scale

            GL.glUniformMatrix4fv(self._uniforms['model_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(model))
            GL.glUniform4fv(self._uniforms['character_color_loc'], 1, np.ascontiguousarray(self._joints_color))

            GL.glDrawArrays(GL.GL_TRIANGLES, self._sphere_mem_info[0], self._sphere_mem_info[1])

    def set_color(self, color):
        self._character_color = color

    def draw(self, frame):
        if self._skeleton is None:
            return

        frame = frame % self._skeleton.frame_count

        # TODO: remove these strings
        self._uniforms = {
            'model_mat_loc': self._shader_program.uniform_location("modelMatrix"),
            'view_mat_loc': self._shader_program.uniform_location("viewMatrix"),
            'proj_mat_loc': self._shader_program.uniform_location("projectionMatrix"),
            'normal_mat_loc': self._shader_program.uniform_location("normalMatrix"),
            'character_color_loc': self._shader_program.uniform_location("characterColor"),
            #'ambient_color_loc': self._shader_program.uniform_location("ambientColor"),
            #'specular_color_loc': self._shader_program.uniform_location("specularColor"),
            #'shininess_loc': self._shader_program.uniform_location("shininess")
        }

        # Setting all shared data
        self._shader_program.bind()

        GL.glUniformMatrix4fv(self._uniforms['view_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(self._view_matrix))
        GL.glUniformMatrix4fv(self._uniforms['proj_mat_loc'], 1, GL.GL_FALSE, np.ascontiguousarray(self._proj_matrix))

        self._skeleton.traverse(frame, self._on_draw_part, self._skeleton_trans)

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
        #vertices, normals = mesh.create_cube_mesh()
        cylinder_mesh = mesh.create_cylinder_mesh(15)
        sphere_mesh   = mesh.create_sphere_mesh(30)

        self._vertices = np.concatenate((cylinder_mesh.vertices, sphere_mesh.vertices))
        self._normals  = np.concatenate((cylinder_mesh.normals , sphere_mesh.normals))

        self._cylinder_mem_info = (0, len(cylinder_mesh.vertices) // 3)
        self._sphere_mem_info   = (len(cylinder_mesh.vertices) // 3, len(sphere_mesh.vertices) // 3)

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
