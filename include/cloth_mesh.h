#pragma once

// ClothMesh: cloth geometry + simulation state (positions/velocities/mass).
//
// Layout contract:
//   GPU device pointers and scalar ints MUST be in the same order for
//   both g++ and nvcc, because pd_solver.cu accesses this struct directly.
//   CPU-only members (Eigen types, std::vector) are guarded by #ifndef __CUDACC__
//   and must come AFTER all scalars to keep offsets consistent.

#ifndef __CUDACC__
#  include <Eigen/Dense>
#  include <string>
#  include <vector>
#endif

struct ClothMesh {
    // ---- GPU: simulation state ----
    float* d_pos       = nullptr;  // current positions  [N*3]
    float* d_vel       = nullptr;  // current velocities [N*3]
    float* d_mass      = nullptr;  // nodal masses       [N]

    // ---- GPU: FEM precomputed geometry ----
    int*   d_tris      = nullptr;  // triangle indices   [T*3]
    float* d_Dm_inv    = nullptr;  // Dm_inv col-major   [T*4] (2×2 per triangle)
    float* d_rest_area = nullptr;  // rest areas         [T]

    // ---- Scalars (visible to both g++ and nvcc) ----
    int num_verts = 0;
    int num_tris  = 0;

#ifndef __CUDACC__
    // ---- CPU geometry (reference configuration) ----
    std::vector<Eigen::Vector3f> rest_pos;   // [N]
    std::vector<Eigen::Vector3i> triangles;  // [T]

    // ---- CPU FEM precomputed ----
    std::vector<Eigen::Matrix2f> Dm_inv;     // [T]  inverse reference edge matrix
    std::vector<float>           rest_area;  // [T]
    std::vector<float>           mass;       // [N]  lumped nodal mass

    // ---- I/O ----
    bool load_obj(const std::string& path);

    // Precompute Dm_inv, rest_area, mass from rest_pos + triangles.
    // Must be called before upload_to_gpu().
    void precompute_rest_state(float density = 0.1f);

    void print_stats() const;
#endif

    // ---- GPU lifecycle ----
    // Upload rest_pos, zero vel, mass, tris, Dm_inv, rest_area to device.
    // Frees any previously allocated buffers first.
    void upload_to_gpu();
    void free_gpu();

    ClothMesh()  = default;
    ~ClothMesh() { free_gpu(); }
    ClothMesh(const ClothMesh&)            = delete;
    ClothMesh& operator=(const ClothMesh&) = delete;
};
