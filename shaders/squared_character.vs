#version 430

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;

in vec3 in_Position;
in vec3 in_Normal;

out vec3 frag_normal;
out vec3 frag_eye_vector;

void main(void)
{
    // Shading is being computed in Eye Space. To make code clear, 'eye' prefix will
    // will be in every vector declaration in this space.
    vec4 es_position = viewMatrix * modelMatrix * vec4(in_Position.xyz, 1.0);

    // mat4 to mat3 conversion has been done in GPU since PyGLM value_ptr function is 
    // still not working for mat3 matrices.
    frag_normal = mat3(normalMatrix) * in_Normal;

    // Assuming light is always located at the camera position. In eye space,
    // ligh position will always be zero.
    frag_eye_vector = es_position.xyz;

    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(in_Position.xyz, 1.0);
}
