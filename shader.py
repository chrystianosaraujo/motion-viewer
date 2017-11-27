import OpenGL.GL as GL
from OpenGL.GL import shaders
import debugger

class ShaderProgram:
    def __init__(self, vert_source_fn, frag_source_fn):
        self._vert_source_fn = vert_source_fn
        self._frag_source_fn = frag_source_fn

        self._attributes = {}
        self._program_id = None

    def bind(self):
        """Activates a shader program.

        Raises a RuntimeError when called before linking/compiling the shader program.
        """

        if self._program_id is None:
            raise RuntimeError("Error while trying to bind a non-compiled shader program.")

        GL.glUseProgram(self._program_id)

    def unbind(self):
        """Deactivates a shader program.

        Raises a RuntimeError when called before linking/compiling the shader program.
        """

        if self._program_id is None:
            raise RuntimeError("Error while trying unbind a non-compiled shader program.")

        GL.glUseProgram(0)

    def compile(self):
        """
        This function compiles and links all shader sources into a shader program.

        After linking a program, no attribute can be bound anymore. Therefore, the
        function bind_attribute_location should not be called.

        This function should only be called once. After being linked, any link call
        will be silently ignored.

        Raises RuntimeErrors when any error occurs while linking or compiling the program.
        """

        if self._program_id is not None:
            return

        self._program_id = GL.glCreateProgram()

        # Compiling shaders
        vert_source_id = self._compile_shader_source(self._vert_source_fn, GL.GL_VERTEX_SHADER)
        frag_source_id = self._compile_shader_source(self._frag_source_fn, GL.GL_FRAGMENT_SHADER)

        # Attaching and linking shaders' source
        GL.glAttachShader(self._program_id, vert_source_id)
        GL.glAttachShader(self._program_id, frag_source_id)

        # Bind Attributes
        for location, name in self._attributes.items():
            GL.glBindAttribLocation(self._program_id, location, name)

        GL.glLinkProgram(self._program_id)

        if GL.glGetProgramiv(self._program_id, GL.GL_LINK_STATUS) == GL.GL_FALSE:
             info = GL.glGetProgramInfoLog(self._program_id)

             GL.glDeleteProgram(self._program_id)
             GL.glDeleteShader(vert_source_id)
             GL.glDeleteShader(frag_source_id)

             raise RuntimeError('Error while linking a shader program: %s' % (info))

        GL.glDeleteShader(vert_source_id)
        GL.glDeleteShader(frag_source_id)

    def bind_attribute(self, location, attrib_name):
        """Binds a vertex attribute. 
        This function should only be used before compiling the shader program.

        Raises a RuntimeError when called before compiling the shader program.
        """

        if self._program_id is not None:
            raise RuntimeError("Error while trying to call bind_attribute "\
                               "for an already compiled shader program.""")

        self._attributes[location] = "attrib_name"

    def attribute_location(self, attrib_name):
        """Returns the location related to the given vertex attribute name. This function should
        only be used after linking and compiling the shader program.

        Raises a RuntimeError when called before linking/compiling the shader program.
        """

        if self._program_id is None:
            raise RuntimeError("Error while trying to call get_attrib_location "\
                               "for a non-compiled shader program.""")

        return GL.glGetAttribLocation(self._program_id, attrib_name)

    def uniform_location(self, uniform_name):
        """Returns the location related to the given uniform name. This function should
        only be used after linking and compiling the shader program.

        Raises a RuntimeError when called before linking/compiling the shader program.
        """

        if self._program_id is None:
            raise RuntimeError("Error while trying to call uniform_location "\
                               "for a non-compiled shader program.""")

        return GL.glGetUniformLocation(self._program_id, uniform_name)


    def _compile_shader_source(self, source_fn, shader_type):
        """This function loads, compiles, and attaches the given shader source into the
        current shader program.

        Parameters:
            source (string): shader filename.
            shader_type (GLType): shader type.
        
        Raises RuntimeError if any error occurs while loading or compiling the shader.
        """

        try:
            with open(source_fn, "r") as f:
                content = f.read()

                shader_id = GL.glCreateShader(shader_type)
                GL.glShaderSource(shader_id, content)
                GL.glCompileShader(shader_id)

                # Check compilation errors
                if GL.glGetShaderiv(shader_id, GL.GL_COMPILE_STATUS) == GL.GL_FALSE:
                    info = GL.glGetShaderInfoLog(shader_id)
                    raise RuntimeError("Error while compiling shader (%s)\n\n%s"%(shader_type, info))
                
                return shader_id

        except IOError:
            raise RuntimeError("Could not load the shader file: %s"%(source_fn))
