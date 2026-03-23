#include "sim_constraints.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <algorithm>

#ifdef CUDA_MS_HAVE_CUDA
#  include <cuda_runtime.h>
#endif

// ============================================================================
// CPU build methods
// ============================================================================

void SimConstraints::build_stretch(const ClothMesh& mesh, float stiffness)
{
    num_tris = mesh.num_tris;
    tri_stretch_k.assign(num_tris, stiffness);
    printf("  SimConstraints: %d triangle stretch constraints (stiffness=%.3f)\n",
           num_tris, stiffness);
}

void SimConstraints::build_bend(const ClothMesh& mesh,
                                const MeshTopology& topo,
                                float stiffness)
{
    bend_quads.clear();
    bend_rest_angles.clear();
    bend_stiffness.clear();

    for (const auto& et : topo.edges) {
        if (et.tri_b == -1) continue;  // boundary edge: no bend

        // Find the vertex opposite to this edge in each adjacent triangle
        auto find_opp = [&](int tri, int v0, int v1) -> int {
            for (int i = 0; i < 3; ++i) {
                int v = mesh.triangles[tri](i);
                if (v != v0 && v != v1) return v;
            }
            return -1;
        };
        const int v2 = find_opp(et.tri_a, et.v0, et.v1);
        const int v3 = find_opp(et.tri_b, et.v0, et.v1);
        if (v2 == -1 || v3 == -1) continue;

        // Compute rest dihedral angle using the wing-vector convention that
        // matches bend_project_kernel: angle between the perpendicular components
        // of v2 and v3 relative to the shared edge axis.
        const Eigen::Vector3f& p0 = mesh.rest_pos[et.v0];
        const Eigen::Vector3f& p1 = mesh.rest_pos[et.v1];
        const Eigen::Vector3f& p2 = mesh.rest_pos[v2];
        const Eigen::Vector3f& p3 = mesh.rest_pos[v3];

        Eigen::Vector3f edge = p1 - p0;
        const float edge_len = edge.norm();
        if (edge_len < 1e-10f) continue;
        Eigen::Vector3f ax = edge / edge_len;

        auto wing = [&](const Eigen::Vector3f& p) -> Eigen::Vector3f {
            const float t = (p - p0).dot(ax);
            return (p - p0) - t * ax;
        };
        Eigen::Vector3f r2 = wing(p2);
        Eigen::Vector3f r3 = wing(p3);
        const float r2_len = r2.norm(), r3_len = r3.norm();
        if (r2_len < 1e-10f || r3_len < 1e-10f) continue;

        const Eigen::Vector3f r2h = r2 / r2_len;
        const Eigen::Vector3f r3h = r3 / r3_len;

        const float cos_t = std::clamp(r2h.dot(r3h), -1.0f, 1.0f);
        const float sin_t = r2h.cross(r3h).dot(ax);
        const float rest_angle = std::atan2(sin_t, cos_t);

        bend_quads.emplace_back(et.v0, et.v1, v2, v3);
        bend_rest_angles.push_back(rest_angle);
        bend_stiffness.push_back(stiffness);
    }

    num_bend_cons = static_cast<int>(bend_quads.size());
    printf("  SimConstraints: %d bend constraints (stiffness=%.4f)\n",
           num_bend_cons, stiffness);
}

// ============================================================================
// Jacobi diagonal precomputation
// ============================================================================

void SimConstraints::precompute_jacobi_diag(const ClothMesh& mesh, float dt)
{
    // System diagonal: diag_i = m_i + h² * Σ_{c∋i} w_c
    //
    // Triangle stretch (Stiefel projection) contribution per triangle t with
    // vertices (v0, v1, v2) and Dm_inv G = [[g00, g01], [g10, g11]]:
    //   diag[v0] += h² * wA * ((g00+g10)² + (g01+g11)²)
    //   diag[v1] += h² * wA * (g00² + g01²)
    //   diag[v2] += h² * wA * (g10² + g11²)
    // where wA = stiffness * rest_area.
    //
    // Bend contribution (v2/v3 wing vertices only — v0/v1 project to current
    // position so they must NOT contribute to the diagonal):
    //   diag[v2] += h² * bend_k
    //   diag[v3] += h² * bend_k

    const int N  = mesh.num_verts;
    const float h2 = dt * dt;

    std::vector<float> diag(N, 0.0f);

    // Mass contribution
    for (int i = 0; i < N; ++i)
        diag[i] = mesh.mass[i];

    // Triangle stretch contribution
    for (int t = 0; t < mesh.num_tris; ++t) {
        if (t >= (int)tri_stretch_k.size()) break;
        const int v0 = mesh.triangles[t](0);
        const int v1 = mesh.triangles[t](1);
        const int v2 = mesh.triangles[t](2);
        const float wA = tri_stretch_k[t] * mesh.rest_area[t];

        // Dm_inv stored col-major: M(row,col)
        const float g00 = mesh.Dm_inv[t](0, 0);
        const float g10 = mesh.Dm_inv[t](1, 0);
        const float g01 = mesh.Dm_inv[t](0, 1);
        const float g11 = mesh.Dm_inv[t](1, 1);

        const float s0 = g00 + g10;
        const float s1 = g01 + g11;
        diag[v0] += h2 * wA * (s0 * s0 + s1 * s1);
        diag[v1] += h2 * wA * (g00 * g00 + g01 * g01);
        diag[v2] += h2 * wA * (g10 * g10 + g11 * g11);
    }

    // Bend contribution (wing vertices only)
    for (int e = 0; e < num_bend_cons; ++e) {
        const float w = bend_stiffness[e];
        if (w == 0.0f) continue;
        diag[bend_quads[e](2)] += h2 * w;  // v2
        diag[bend_quads[e](3)] += h2 * w;  // v3
    }

#ifdef CUDA_MS_HAVE_CUDA
    if (d_jacobi_diag) { cudaFree(d_jacobi_diag); d_jacobi_diag = nullptr; }
    cudaMalloc((void**)&d_jacobi_diag, N * sizeof(float));
    cudaMemcpy(d_jacobi_diag, diag.data(), N * sizeof(float), cudaMemcpyHostToDevice);
#endif
}

// ============================================================================
// GPU lifecycle
// ============================================================================

void SimConstraints::upload_to_gpu()
{
#ifdef CUDA_MS_HAVE_CUDA
    free_gpu();

    auto alloc_copy = [](void** dst, const void* src, size_t bytes) {
        cudaMalloc(dst, bytes);
        cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice);
    };

    if (!tri_stretch_k.empty()) {
        alloc_copy((void**)&d_tri_stretch_k,
                   tri_stretch_k.data(), num_tris * sizeof(float));
    }

    // d_jacobi_diag is uploaded by precompute_jacobi_diag(), not here.

    if (num_bend_cons > 0) {
        std::vector<int> h_quads(num_bend_cons * 4);
        for (int i = 0; i < num_bend_cons; ++i) {
            h_quads[i * 4 + 0] = bend_quads[i](0);
            h_quads[i * 4 + 1] = bend_quads[i](1);
            h_quads[i * 4 + 2] = bend_quads[i](2);
            h_quads[i * 4 + 3] = bend_quads[i](3);
        }
        alloc_copy((void**)&d_bend_quads, h_quads.data(),
                   num_bend_cons * 4 * sizeof(int));
        alloc_copy((void**)&d_bend_rest, bend_rest_angles.data(),
                   num_bend_cons * sizeof(float));
        alloc_copy((void**)&d_bend_k, bend_stiffness.data(),
                   num_bend_cons * sizeof(float));
    }
#else
    fprintf(stderr, "SimConstraints::upload_to_gpu: CUDA not compiled in — skipping.\n");
#endif
}

void SimConstraints::free_gpu()
{
#ifdef CUDA_MS_HAVE_CUDA
    auto safe_free = [](void* p) { if (p) cudaFree(p); };
    safe_free(d_tri_stretch_k); d_tri_stretch_k = nullptr;
    safe_free(d_jacobi_diag);   d_jacobi_diag   = nullptr;
    safe_free(d_bend_quads);    d_bend_quads     = nullptr;
    safe_free(d_bend_rest);     d_bend_rest      = nullptr;
    safe_free(d_bend_k);        d_bend_k         = nullptr;
#endif
}
