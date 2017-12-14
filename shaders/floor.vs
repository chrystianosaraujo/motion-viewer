#version 430

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

in vec3 in_Position;
in vec3 in_Normal;
in vec2 in_TexCoord;

out vec2 texcoord;

void main(void)
{
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(in_Position.xyz, 1.0);
    texcoord = in_TexCoord; 
}
