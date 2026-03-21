#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Arcball orbit camera
struct OrbitCamera {
    float distance = 5.0f;
    float yaw = 0.0f;      // Horizontal rotation (degrees)
    float pitch = 30.0f;   // Vertical rotation (degrees), limited to -89~89
    glm::vec3 target = {0, 0, 0};
    float fov = 45.0f;
    float near_plane = 0.1f;
    float far_plane = 100.0f;

    // Get view matrix
    glm::mat4 get_view_matrix() const;

    // Get projection matrix
    glm::mat4 get_proj_matrix(float aspect) const;

    // Rotate camera
    void rotate(float delta_yaw, float delta_pitch);

    // Zoom camera
    void zoom(float delta_distance);

    // Get camera position in world space
    glm::vec3 get_position() const;
};
