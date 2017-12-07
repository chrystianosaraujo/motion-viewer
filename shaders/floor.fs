#version 430

uniform vec4 ambientColor;
uniform vec4 diffuseColor;

out vec4 fs_out_color;

void main(void)
{
	fs_out_color = vec4(1.0, 1.0, 1.0, 1.0);
}
