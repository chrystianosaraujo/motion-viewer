#version 430

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

in vec3 in_Position;
in vec3 in_Normal;

out vec3 frag_normal;
out vec3 frag_eye_vector;
out vec3 fragVert;

void main() {
    frag_normal = in_Normal;
    fragVert    = in_Position;

    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(in_Position.xyz, 1.0);
}
