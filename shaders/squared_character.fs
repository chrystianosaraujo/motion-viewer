#version 430

uniform vec4 ambientColor;
uniform vec4 characterColor;

uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

in vec3 frag_normal;
in vec3 fragVert;

out vec4 fs_out_color;


void main(void)
{
  // Camera space origin
  vec3 lightPos = vec3(0.0, 0.0, 0.0);

  mat3 normalMatrix = transpose(inverse(mat3(viewMatrix * modelMatrix)));
  vec3 normal = normalize(normalMatrix * frag_normal);

  vec3 fragPosition = vec3(viewMatrix * modelMatrix * vec4(fragVert, 1));

  vec3 surfaceToLight = normalize(lightPos - fragPosition);

  float brightness = dot(normal, surfaceToLight);
  brightness = clamp(brightness, 0, 1);

  fs_out_color = (brightness * characterColor + ambientColor);
}
