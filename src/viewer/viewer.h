#pragma once

#include "cloth_mesh.h"
#include <memory>

class ClothViewerImpl;

class ClothViewer {
public:
    ClothViewer();
    ~ClothViewer();

    // Initialize window and OpenGL context
    bool init(int width = 1280, int height = 720, const char* title = "cuda-ms");

    // Upload mesh data to GPU (call once after mesh is generated)
    void upload_mesh(const ClothMesh& mesh);

    // Check if window should close
    bool should_close() const;

    // Begin frame: poll events + start ImGui frame
    void begin_frame();

    // Render mesh and ImGui
    void render(const ClothMesh& mesh);

    // End frame: swap buffers
    void end_frame();

    // Camera controls
    void set_camera_distance(float dist);
    void rotate_camera(float delta_yaw, float delta_pitch);

private:
    std::unique_ptr<ClothViewerImpl> impl_;
};
