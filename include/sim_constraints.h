#pragma once

// SimConstraints: PD constraint data and the precomputed Jacobi system diagonal.
//
// Holds GPU device pointers for:
//   - per-triangle stretch stiffness  (d_tri_stretch_k)
//   - Jacobi system diagonal          (d_jacobi_diag)
//   - bend constraint quads/rest/k    (d_bend_quads / d_bend_rest / d_bend_k)
//
// CPU build methods are guarded by #ifndef __CUDACC__ (they use Eigen/std types).
// GPU pointers and scalar counts are always visible to nvcc.

#ifndef __CUDACC__
#  include "cloth_mesh.h"
#  include "mesh_topology.h"
#  include <Eigen/Dense>
#  include <vector>
#endif

struct SimConstraints {
    // ---- GPU: triangle stretch ----
    float* d_tri_stretch_k = nullptr;  // [T]  per-triangle stiffness

    // ---- GPU: Jacobi system diagonal  M_ii + h² Σ w_c ----
    float* d_jacobi_diag   = nullptr;  // [N]

    // ---- GPU: bend constraints ----
    int*   d_bend_quads    = nullptr;  // [E_bend * 4]  (v0,v1,v2,v3)
    float* d_bend_rest     = nullptr;  // [E_bend]      rest dihedral angles
    float* d_bend_k        = nullptr;  // [E_bend]      stiffness weights

    // ---- Scalars ----
    int num_tris      = 0;
    int num_bend_cons = 0;

#ifndef __CUDACC__
    // ---- CPU constraint data ----
    std::vector<float>           tri_stretch_k;   // [T]
    std::vector<Eigen::Vector4i> bend_quads;      // [E_bend]  (v0,v1,v2,v3)
    std::vector<float>           bend_rest_angles; // [E_bend]
    std::vector<float>           bend_stiffness;   // [E_bend]

    // Set per-triangle stretch stiffness (uniform).
    void build_stretch(const ClothMesh& mesh, float stiffness);

    // Build bend constraints from inner edges.
    // Requires mesh.rest_pos and a fully-built MeshTopology.
    void build_bend(const ClothMesh& mesh,
                    const MeshTopology& topo,
                    float stiffness);

    // Compute Jacobi diagonal from current CPU constraint data and upload to GPU.
    // Must be called after build_stretch (and build_bend if used), and before
    // any PDSolver::step() calls.
    void precompute_jacobi_diag(const ClothMesh& mesh, float dt);
#endif

    // ---- GPU lifecycle ----
    // Upload all constraint data to device.
    void upload_to_gpu();
    void free_gpu();

    SimConstraints()  = default;
    ~SimConstraints() { free_gpu(); }
    SimConstraints(const SimConstraints&)            = delete;
    SimConstraints& operator=(const SimConstraints&) = delete;
};
