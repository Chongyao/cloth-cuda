#pragma once

#include "glad/glad.h"
#include <glm/glm.hpp>
#include <string>

// Simple shader loader
class Shader {
public:
    Shader() = default;
    ~Shader();

    // Disable copy
    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;

    // Enable move
    Shader(Shader&& other) noexcept;
    Shader& operator=(Shader&& other) noexcept;

    // Load shaders from source strings
    bool load_from_source(const std::string& vertex_source,
                          const std::string& fragment_source);

    // Load shaders from files
    bool load_from_file(const std::string& vertex_path,
                        const std::string& fragment_path);

    // Use shader
    void use() const;

    // Set uniforms
    void set_bool(const std::string& name, bool value) const;
    void set_int(const std::string& name, int value) const;
    void set_float(const std::string& name, float value) const;
    void set_vec3(const std::string& name, const glm::vec3& value) const;
    void set_vec3(const std::string& name, float x, float y, float z) const;
    void set_mat4(const std::string& name, const glm::mat4& value) const;

    // Check if shader is valid
    bool is_valid() const { return program_id_ != 0; }

    GLuint get_id() const { return program_id_; }

private:
    GLuint program_id_ = 0;

    bool compile_shader(GLuint shader, const std::string& source);
    bool link_program(GLuint vertex_shader, GLuint fragment_shader);
    GLint get_uniform_location(const std::string& name) const;
};
