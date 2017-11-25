#version 430

in vec3 in_Position;
out vec3 vs_out_color;

void main(void)
{
    vs_out_color = vec3(1.0, 0.0, 0.0);
    gl_Position = vec4(in_Position.xyz, 1.0);
}
