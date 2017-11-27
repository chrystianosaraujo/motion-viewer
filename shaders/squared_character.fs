#version 430

uniform vec4 ambientColor;
uniform vec4 diffuseColor;

in vec3 frag_normal;
in vec3 frag_eye_vector;

out vec4 fs_out_color;

void main(void)
{
  vec3 normal  = normalize(frag_normal);
  vec3 eye_vec = normalize(frag_eye_vector);

  // Diffuse intensity
  float kd = max(dot(normal, eye_vec), 0.0);
  fs_out_color = max(kd * diffuseColor, ambientColor);

  // TODO: implement full phong-bliin model
}
