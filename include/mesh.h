#pragma once

#include <vector>
#include <string>

// Eigen is only included when compiled by g++/clang (not nvcc).
// Any .cu file must NOT include this header directly.
#ifndef __CUDACC__
#  include <Eigen/Dense>
#endif

struct ClothMesh {
    // ---- CPU raw data ----
#ifndef __CUDACC__
    std::vector<Eigen::Vector3f> rest_pos;   // reference-config vertices [N]
    std::vector<Eigen::Vector3i> triangles;  // triangle vertex indices    [T]

    // ---- CPU precomputed ----
    std::vector<Eigen::Matrix2f> Dm_inv;     // inverse reference edge matrix [T]

    // ---- Inner-edge topology (for bending) ----
    // Each entry (v0, v1, v2, v3): v0-v1 is the shared edge,
    // v2 is the opposite vertex in triangle A, v3 in triangle B.
    std::vector<Eigen::Vector4i> inner_edges;
#endif
    std::vector<float>           rest_area;  // triangle areas in rest config  [T]
    std::vector<float>           mass;       // lumped nodal masses             [N]

    // ---- GPU mirrors (device pointers, null when CUDA unavailable) ----
    float* d_pos        = nullptr;  // current positions  [N*3]
    float* d_vel        = nullptr;  // current velocities [N*3]
    int*   d_tris       = nullptr;  // triangle indices   [T*3]
    float* d_Dm_inv     = nullptr;  // Dm_inv flattened   [T*4] (col-major 2×2)
    float* d_rest_area  = nullptr;  // rest areas         [T]
    float* d_mass       = nullptr;  // nodal masses       [N]
    int*   d_inner_edges = nullptr; // inner edges        [IE*4]

    int num_verts      = 0;
    int num_tris       = 0;
    int num_inner_edges = 0;

    // ---- Lifecycle ----
    ClothMesh()  = default;
    ~ClothMesh();

    ClothMesh(const ClothMesh&)            = delete;
    ClothMesh& operator=(const ClothMesh&) = delete;

    // ---- Operations (mesh.cpp, compiled by g++) ----
    bool load_obj(const std::string& path);
    void precompute_rest_state(float density = 0.1f);
    void build_inner_edges();
    void upload_to_gpu();   // no-op when CUDA not available
    void free_gpu();

    void print_stats() const;
};
