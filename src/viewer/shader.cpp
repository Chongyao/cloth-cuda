#include "shader.h"
#include "glad/glad.h"
#include <fstream>
#include <sstream>
#include <iostream>

Shader::~Shader() {
    if (program_id_ != 0) {
        glDeleteProgram(program_id_);
    }
}

Shader::Shader(Shader&& other) noexcept
    : program_id_(other.program_id_) {
    other.program_id_ = 0;
}

Shader& Shader::operator=(Shader&& other) noexcept {
    if (this != &other) {
        if (program_id_ != 0) {
            glDeleteProgram(program_id_);
        }
        program_id_ = other.program_id_;
        other.program_id_ = 0;
    }
    return *this;
}

bool Shader::load_from_source(const std::string& vertex_source,
                               const std::string& fragment_source) {
    // Create shaders
    GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);

    if (!compile_shader(vertex_shader, vertex_source)) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return false;
    }

    if (!compile_shader(fragment_shader, fragment_source)) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return false;
    }

    if (!link_program(vertex_shader, fragment_shader)) {
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return false;
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return true;
}

bool Shader::load_from_file(const std::string& vertex_path,
                            const std::string& fragment_path) {
    std::string vertex_source;
    std::string fragment_source;

    // Read vertex shader
    {
        std::ifstream file(vertex_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open vertex shader: " << vertex_path << std::endl;
            return false;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        vertex_source = buffer.str();
    }

    // Read fragment shader
    {
        std::ifstream file(fragment_path);
        if (!file.is_open()) {
            std::cerr << "Failed to open fragment shader: " << fragment_path << std::endl;
            return false;
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        fragment_source = buffer.str();
    }

    return load_from_source(vertex_source, fragment_source);
}

bool Shader::compile_shader(GLuint shader, const std::string& source) {
    const char* source_cstr = source.c_str();
    glShaderSource(shader, 1, &source_cstr, nullptr);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetShaderInfoLog(shader, 512, nullptr, info_log);
        std::cerr << "Shader compilation failed:\n" << info_log << std::endl;
        return false;
    }
    return true;
}

bool Shader::link_program(GLuint vertex_shader, GLuint fragment_shader) {
    if (program_id_ != 0) {
        glDeleteProgram(program_id_);
    }

    program_id_ = glCreateProgram();
    glAttachShader(program_id_, vertex_shader);
    glAttachShader(program_id_, fragment_shader);
    glLinkProgram(program_id_);

    GLint success;
    glGetProgramiv(program_id_, GL_LINK_STATUS, &success);
    if (!success) {
        char info_log[512];
        glGetProgramInfoLog(program_id_, 512, nullptr, info_log);
        std::cerr << "Shader linking failed:\n" << info_log << std::endl;
        glDeleteProgram(program_id_);
        program_id_ = 0;
        return false;
    }
    return true;
}

void Shader::use() const {
    if (program_id_ != 0) {
        glUseProgram(program_id_);
    }
}

GLint Shader::get_uniform_location(const std::string& name) const {
    if (program_id_ == 0) return -1;
    return glGetUniformLocation(program_id_, name.c_str());
}

void Shader::set_bool(const std::string& name, bool value) const {
    glUniform1i(get_uniform_location(name), value ? 1 : 0);
}

void Shader::set_int(const std::string& name, int value) const {
    glUniform1i(get_uniform_location(name), value);
}

void Shader::set_float(const std::string& name, float value) const {
    glUniform1f(get_uniform_location(name), value);
}

void Shader::set_vec3(const std::string& name, const glm::vec3& value) const {
    glUniform3fv(get_uniform_location(name), 1, &value[0]);
}

void Shader::set_vec3(const std::string& name, float x, float y, float z) const {
    glUniform3f(get_uniform_location(name), x, y, z);
}

void Shader::set_mat4(const std::string& name, const glm::mat4& value) const {
    glUniformMatrix4fv(get_uniform_location(name), 1, GL_FALSE, &value[0][0]);
}
