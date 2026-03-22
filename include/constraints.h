#pragma once
#include "mesh.h"
#include <vector>

struct Constraints {
    std::vector<int>   pinned_indices;    // fixed vertex indices
    std::vector<float> target_positions; // desired positions [num_verts * 3]

    // --- Preset boundary conditions ---

    // Fix the first row (indices 0 .. ncols-1)
    void pin_top_row(const ClothMesh& mesh, int ncols);

    // Fix the two top corners (index 0 and ncols-1)
    void pin_corners(const ClothMesh& mesh, int ncols);

    // Fix an arbitrary list of vertex indices
    void set_from_list(const std::vector<int>& indices,
                       const ClothMesh& mesh);

    // Copy current rest positions of pinned vertices into target_positions
    void apply_to_mesh(ClothMesh& mesh) const;

    void clear();
    void print_stats() const;

    // ---- GPU support ----
    int* d_pinned_indices = nullptr;      // device pointer [num_pinned]
    float* d_target_pos = nullptr;        // device pointer [num_pinned * 3]
    int num_pinned = 0;

    void upload_to_gpu();
    void free_gpu();
    void apply_gpu(float* d_pos, float* d_vel) const;  // set pinned vertices
};
