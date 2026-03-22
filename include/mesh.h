#pragma once

#include <vector>
#include <string>

// Eigen is only included when compiled by g++/clang (not nvcc).
// Any .cu file must NOT include this header directly.
#ifndef __CUDACC__
#  include <Eigen/Dense>
#endif

// ClothMesh - shared between C++ and CUDA
// IMPORTANT: Keep memory layout consistent between .cpp and .cu files!
// - GPU pointers and scalar members must be in the SAME ORDER for both
// - CPU-only members go at the end, guarded by #ifndef __CUDACC__

struct ClothMesh {
    // ---- GPU mirrors (device pointers, null when CUDA unavailable) ----
    // These MUST be at the same offset for .cpp and .cu
    float* d_pos        = nullptr;  // current positions  [N*3]
    float* d_vel        = nullptr;  // current velocities [N*3]
    float* d_prev_pos   = nullptr;  // previous positions for Chebyshev [N*3]
    int*   d_tris       = nullptr;  // triangle indices   [T*3]
    float* d_Dm_inv     = nullptr;  // Dm_inv flattened   [T*4] (col-major 2×2)
    float* d_rest_area  = nullptr;  // rest areas         [T]
    float* d_mass       = nullptr;  // nodal masses       [N]
    int*   d_inner_edges = nullptr; // inner edges        [IE*4]

    // ---- PD constraint GPU data ----
    int*   d_stretch_edges   = nullptr;  // [E_stretch*2] vertex indices
    float* d_stretch_rest    = nullptr;  // [E_stretch] rest lengths
    float* d_stretch_k       = nullptr;  // [E_stretch] stiffness weights
    float* d_jacobi_diag     = nullptr;  // [N] precomputed M_ii + h²*(ΣwAᵀA)_ii

    // Bend constraints (Phase 4)
    int*   d_bend_quads      = nullptr;  // [E_bend*4] vertex indices (v0,v1,v2,v3)
    float* d_bend_rest       = nullptr;  // [E_bend] rest angles
    float* d_bend_k          = nullptr;  // [E_bend] stiffness weights

    // ---- Scalar members (same for .cpp and .cu) ----
    int num_verts      = 0;
    int num_tris       = 0;
    int num_inner_edges = 0;
    int num_stretch_cons = 0;  // stretch constraints count

#ifndef __CUDACC__
    // ---- CPU raw data (ONLY for .cpp files) ----
    std::vector<Eigen::Vector3f> rest_pos;   // reference-config vertices [N]
    std::vector<Eigen::Vector3i> triangles;  // triangle vertex indices    [T]

    // ---- CPU precomputed ----
    std::vector<Eigen::Matrix2f> Dm_inv;     // inverse reference edge matrix [T]

    // ---- Inner-edge topology (for bending) ----
    std::vector<Eigen::Vector4i> inner_edges;

    // ---- Stretch constraints (for PD) ----
    std::vector<Eigen::Vector4f> stretch_constraints;

    // ---- Bend constraints (for PD, Phase 4) ----
    std::vector<float> bend_rest_angles;
    std::vector<float> bend_stiffness;

    // ---- CPU data (std::vector compatible) ----
    std::vector<float> rest_area;
    std::vector<float> mass;
#endif

    // ---- Lifecycle ----
    ClothMesh()  = default;
    ~ClothMesh();

    ClothMesh(const ClothMesh&)            = delete;
    ClothMesh& operator=(const ClothMesh&) = delete;

    // ---- Operations ----
#ifndef __CUDACC__
    bool load_obj(const std::string& path);
    void precompute_rest_state(float density = 0.1f);
    void build_inner_edges();
    void build_stretch_constraints(float stiffness = 1.0f);
    void build_bend_constraints(float stiffness = 0.01f);
    void precompute_jacobi_diag(float dt, float constraint_wt = 1.0f);
#endif
    void upload_to_gpu();   // no-op when CUDA not available
    void free_gpu();

#ifndef __CUDACC__
    void print_stats() const;
#endif
};
