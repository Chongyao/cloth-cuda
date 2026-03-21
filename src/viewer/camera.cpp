#include "camera.h"
#include <algorithm>

glm::mat4 OrbitCamera::get_view_matrix() const {
    // Convert spherical coordinates to Cartesian
    float yaw_rad = glm::radians(yaw);
    float pitch_rad = glm::radians(pitch);

    float x = distance * cos(pitch_rad) * sin(yaw_rad);
    float y = distance * sin(pitch_rad);
    float z = distance * cos(pitch_rad) * cos(yaw_rad);

    glm::vec3 position = target + glm::vec3(x, y, z);
    return glm::lookAt(position, target, glm::vec3(0, 1, 0));
}

glm::mat4 OrbitCamera::get_proj_matrix(float aspect) const {
    return glm::perspective(glm::radians(fov), aspect, near_plane, far_plane);
}

void OrbitCamera::rotate(float delta_yaw, float delta_pitch) {
    yaw += delta_yaw;
    pitch += delta_pitch;

    // Clamp pitch to avoid gimbal lock
    pitch = std::clamp(pitch, -89.0f, 89.0f);

    // Normalize yaw to 0-360
    while (yaw >= 360.0f) yaw -= 360.0f;
    while (yaw < 0.0f) yaw += 360.0f;
}

void OrbitCamera::zoom(float delta_distance) {
    distance = std::max(0.1f, distance - delta_distance);
}

glm::vec3 OrbitCamera::get_position() const {
    float yaw_rad = glm::radians(yaw);
    float pitch_rad = glm::radians(pitch);

    float x = distance * cos(pitch_rad) * sin(yaw_rad);
    float y = distance * sin(pitch_rad);
    float z = distance * cos(pitch_rad) * cos(yaw_rad);

    return target + glm::vec3(x, y, z);
}
