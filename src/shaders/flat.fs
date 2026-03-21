#version 410 core
out vec4 FragColor;
uniform vec3 meshColor;
void main() {
    FragColor = vec4(meshColor, 1.0);
}
