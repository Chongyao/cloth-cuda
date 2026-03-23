#pragma once

// SimConstraints: PD constraint data and the precomputed Jacobi system diagonal.
//
// Stretch:  Stiefel manifold projection (SVD of deformation gradient F → R).
//           One constraint per triangle; geometry stored in ClothMesh.
//
// Bend:     Cotangent-weighted quadratic bending (Discrete Shells / DiffCloth).
//           For hinge (v0,v1,v2,v3) with shared edge (v0,v1):
//             e     = Σ w_i · x_i          (discrete curvature vector)
//             p_proj = e/|e| · n_rest       (project to sphere of rest curvature)
//           Weights: w0 = cot02+cot03, w1 = cot12+cot13,
//                    w2 = -(cot02+cot12), w3 = -(cot03+cot13)
//           n_rest = |Σ w_i · X_i|_rest   (rest curvature magnitude; 0 for flat cloth)
//
// Jacobi diagonal:
//   stretch:  diag[vi] += h² · wA · ‖G_col_i‖²
//   bend:     diag[vi] += h² · k  · w_i²
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
    float* d_tri_stretch_k = nullptr;  // [T]      per-triangle stiffness

    // ---- GPU: Jacobi system diagonal  M_ii + h² Σ w_c ----
    float* d_jacobi_diag   = nullptr;  // [N]

    // ---- GPU: cotangent bend constraints ----
    int*   d_bend_quads    = nullptr;  // [E_bend*4]  vertex indices (v0,v1,v2,v3)
    float* d_bend_w        = nullptr;  // [E_bend*4]  cotangent weights per vertex
    float* d_bend_n        = nullptr;  // [E_bend]    rest curvature norm |Σ w_i X_i|
    float* d_bend_k        = nullptr;  // [E_bend]    stiffness per constraint

    // ---- Scalars ----
    int num_tris      = 0;
    int num_bend_cons = 0;

#ifndef __CUDACC__
    // ---- CPU constraint data ----
    std::vector<float>           tri_stretch_k;  // [T]
    std::vector<Eigen::Vector4i> bend_quads;     // [E_bend]  (v0,v1,v2,v3)
    std::vector<float>           bend_w;         // [E_bend*4] cotangent weights
    std::vector<float>           bend_n;         // [E_bend]   rest curvature norms
    std::vector<float>           bend_stiffness; // [E_bend]

    // Set per-triangle stretch stiffness (uniform).
    void build_stretch(const ClothMesh& mesh, float stiffness);

    // Build cotangent-weighted bend constraints from inner edges.
    // Uses DiffCloth / Discrete Shells cotangent weight formulation.
    // Requires mesh.rest_pos, mesh.triangles, and a fully-built MeshTopology.
    void build_bend(const ClothMesh& mesh,
                    const MeshTopology& topo,
                    float stiffness);

    // Compute Jacobi diagonal from current CPU constraint data and upload to GPU.
    // Call after build_stretch / build_bend, before upload_to_gpu.
    void precompute_jacobi_diag(const ClothMesh& mesh, float dt);
#endif

    // ---- GPU lifecycle ----
    void upload_to_gpu();
    void free_gpu();

    SimConstraints()  = default;
    ~SimConstraints() { free_gpu(); }
    SimConstraints(const SimConstraints&)            = delete;
    SimConstraints& operator=(const SimConstraints&) = delete;
};
