#version 430

uniform vec4 ambientColor;
uniform vec4 diffuseColor;

out vec4 fs_out_color;

in vec2 texcoord;

uniform sampler2D color_map;

void main(void)
{
	fs_out_color = texture(color_map, texcoord);
}
