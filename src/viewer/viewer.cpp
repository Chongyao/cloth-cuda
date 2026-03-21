#include "viewer.h"
#include "camera.h"
#include "shader.h"
#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

class ClothViewerImpl {
public:
    GLFWwindow* window = nullptr;
    Shader shader;
    OrbitCamera camera;

    // OpenGL objects
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;

    // Mesh stats
    int num_verts_ = 0;
    int num_tris_ = 0;
    int num_inner_edges_ = 0;

    // UI state
    bool show_wireframe_ = false;
    bool enable_culling_ = true;
    float mesh_color_[3] = {0.8f, 0.6f, 0.4f};
    float clear_color_[3] = {0.2f, 0.2f, 0.2f};

    // Mouse state
    bool mouse_left_down_ = false;
    double last_mouse_x_ = 0;
    double last_mouse_y_ = 0;

    ~ClothViewerImpl() {
        cleanup_gl();
    }

    void cleanup_gl() {
        if (vao) { glDeleteVertexArrays(1, &vao); vao = 0; }
        if (vbo) { glDeleteBuffers(1, &vbo); vbo = 0; }
        if (ebo) { glDeleteBuffers(1, &ebo); ebo = 0; }
    }

    void setup_gl() {
        // Enable depth testing
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        // Enable backface culling
        if (enable_culling_) {
            glEnable(GL_CULL_FACE);
            glCullFace(GL_BACK);
            glFrontFace(GL_CCW);
        }

        // Set clear color
        glClearColor(clear_color_[0], clear_color_[1], clear_color_[2], 1.0f);
    }

    void create_buffers() {
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glGenBuffers(1, &ebo);
    }

    void handle_mouse_input() {
        if (!window) return;

        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        if (mouse_left_down_) {
            float dx = static_cast<float>(xpos - last_mouse_x_);
            float dy = static_cast<float>(ypos - last_mouse_y_);

            // Rotate camera: left drag
            const float sensitivity = 0.5f;
            camera.rotate(dx * sensitivity, -dy * sensitivity);
        }

        last_mouse_x_ = xpos;
        last_mouse_y_ = ypos;
    }

    void draw_mesh(const ClothMesh& mesh);
    void draw_ui(const ClothMesh& mesh);
};

// Static callbacks
static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    ClothViewerImpl* impl = static_cast<ClothViewerImpl*>(glfwGetWindowUserPointer(window));
    if (!impl) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        impl->mouse_left_down_ = (action == GLFW_PRESS);
    }
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    ClothViewerImpl* impl = static_cast<ClothViewerImpl*>(glfwGetWindowUserPointer(window));
    if (!impl) return;

    // Zoom on scroll
    impl->camera.zoom(static_cast<float>(yoffset) * 0.5f);
}

ClothViewer::ClothViewer() : impl_(std::make_unique<ClothViewerImpl>()) {}

ClothViewer::~ClothViewer() = default;

bool ClothViewer::init(int width, int height, const char* title) {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }

    // Set OpenGL version (4.1 Core)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Create window
    impl_->window = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!impl_->window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(impl_->window);
    glfwSwapInterval(1); // Enable vsync

    // Load OpenGL functions via GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }

    // Set up viewport
    glViewport(0, 0, width, height);
    glfwSetFramebufferSizeCallback(impl_->window, framebuffer_size_callback);

    // Set user pointer for callbacks
    glfwSetWindowUserPointer(impl_->window, impl_.get());
    glfwSetMouseButtonCallback(impl_->window, mouse_button_callback);
    glfwSetScrollCallback(impl_->window, scroll_callback);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(impl_->window, true);
    ImGui_ImplOpenGL3_Init("#version 410 core");

    // Setup OpenGL state
    impl_->setup_gl();
    impl_->create_buffers();

    return true;
}

void ClothViewer::upload_mesh(const ClothMesh& mesh) {
    if (!impl_->window || impl_->vao == 0) return;

    // Store stats
    impl_->num_verts_ = mesh.num_verts;
    impl_->num_tris_ = mesh.num_tris;
    impl_->num_inner_edges_ = mesh.num_inner_edges;

    // Flatten vertex positions
    std::vector<float> vertices;
    vertices.reserve(mesh.num_verts * 3);
    for (const auto& p : mesh.rest_pos) {
        vertices.push_back(p.x());
        vertices.push_back(p.y());
        vertices.push_back(p.z());
    }

    // Flatten triangle indices
    std::vector<unsigned int> indices;
    indices.reserve(mesh.num_tris * 3);
    for (const auto& t : mesh.triangles) {
        indices.push_back(static_cast<unsigned int>(t.x()));
        indices.push_back(static_cast<unsigned int>(t.y()));
        indices.push_back(static_cast<unsigned int>(t.z()));
    }

    // Upload to GPU
    glBindVertexArray(impl_->vao);

    glBindBuffer(GL_ARRAY_BUFFER, impl_->vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, impl_->ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

    // Set vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);

    // Center camera on mesh
    float mesh_size = 0.0f;
    for (const auto& p : mesh.rest_pos) {
        mesh_size = std::max(mesh_size, std::max({std::abs(p.x()), std::abs(p.y()), std::abs(p.z())}));
    }
    if (mesh_size > 0) {
        impl_->camera.distance = mesh_size * 3.0f;
        impl_->camera.target = glm::vec3(0, 0, 0);
    }
}

bool ClothViewer::should_close() const {
    return impl_->window && glfwWindowShouldClose(impl_->window);
}

void ClothViewer::begin_frame() {
    glfwPollEvents();

    // Handle mouse input for camera rotation
    impl_->handle_mouse_input();

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ClothViewer::render(const ClothMesh& mesh) {
    // Clear screen
    glClearColor(impl_->clear_color_[0], impl_->clear_color_[1], impl_->clear_color_[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw mesh
    impl_->draw_mesh(mesh);

    // Draw UI
    impl_->draw_ui(mesh);
}

void ClothViewer::end_frame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(impl_->window);
}

void ClothViewer::set_camera_distance(float dist) {
    impl_->camera.distance = std::max(0.1f, dist);
}

void ClothViewer::rotate_camera(float delta_yaw, float delta_pitch) {
    impl_->camera.rotate(delta_yaw, delta_pitch);
}

void ClothViewerImpl::draw_mesh(const ClothMesh& mesh) {
    if (vao == 0 || num_tris_ == 0) return;

    // Build a simple shader inline if not loaded
    static bool shader_loaded = false;
    if (!shader_loaded) {
        const char* vertex_source = R"(
            #version 410 core
            layout(location = 0) in vec3 aPos;
            uniform mat4 MVP;
            void main() {
                gl_Position = MVP * vec4(aPos, 1.0);
            }
        )";

        const char* fragment_source = R"(
            #version 410 core
            out vec4 FragColor;
            uniform vec3 meshColor;
            void main() {
                FragColor = vec4(meshColor, 1.0);
            }
        )";

        shader_loaded = shader.load_from_source(vertex_source, fragment_source);
    }

    if (!shader.is_valid()) return;

    shader.use();

    // Get window size for aspect ratio
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    float aspect = width > 0 && height > 0 ? static_cast<float>(width) / height : 1.0f;

    // Compute MVP matrix
    glm::mat4 view = camera.get_view_matrix();
    glm::mat4 proj = camera.get_proj_matrix(aspect);
    glm::mat4 mvp = proj * view;  // Model is identity (mesh is in world space)

    shader.set_mat4("MVP", mvp);
    shader.set_vec3("meshColor", mesh_color_[0], mesh_color_[1], mesh_color_[2]);

    // Enable/disable face culling
    if (enable_culling_) {
        glEnable(GL_CULL_FACE);
    } else {
        glDisable(GL_CULL_FACE);
    }

    glBindVertexArray(vao);

    // Draw filled triangles
    if (!show_wireframe_) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawElements(GL_TRIANGLES, num_tris_ * 3, GL_UNSIGNED_INT, 0);
    }

    // Draw wireframe overlay
    if (show_wireframe_) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glEnable(GL_POLYGON_OFFSET_LINE);
        glPolygonOffset(-1.0f, -1.0f);
        shader.set_vec3("meshColor", 0.0f, 0.0f, 0.0f);  // Black wireframe
        glDrawElements(GL_TRIANGLES, num_tris_ * 3, GL_UNSIGNED_INT, 0);
        glDisable(GL_POLYGON_OFFSET_LINE);
    }

    glBindVertexArray(0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void ClothViewerImpl::draw_ui(const ClothMesh& mesh) {
    ImGui::Begin("Mesh Stats");
    ImGui::Text("Vertices: %d", num_verts_);
    ImGui::Text("Triangles: %d", num_tris_);
    ImGui::Text("Inner edges: %d", num_inner_edges_);

    ImGui::Separator();
    ImGui::Checkbox("Wireframe", &show_wireframe_);
    ImGui::Checkbox("Backface culling", &enable_culling_);
    ImGui::ColorEdit3("Mesh color", mesh_color_);
    ImGui::ColorEdit3("Background", clear_color_);

    ImGui::Separator();
    ImGui::Text("Camera distance: %.2f", camera.distance);
    ImGui::Text("Yaw: %.1f, Pitch: %.1f", camera.yaw, camera.pitch);

    ImGui::End();
}
