#include "sim_constraints.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <algorithm>

#ifdef CUDA_MS_HAVE_CUDA
#  include <cuda_runtime.h>
#endif

// ============================================================================
// CPU build: stretch
// ============================================================================

void SimConstraints::build_stretch(const ClothMesh& mesh, float stiffness)
{
    num_tris = mesh.num_tris;
    tri_stretch_k.assign(num_tris, stiffness);
    printf("  SimConstraints: %d triangle stretch constraints (stiffness=%.3f)\n",
           num_tris, stiffness);
}

// ============================================================================
// CPU build: cotangent-weighted bend (Discrete Shells / DiffCloth)
//
// For each inner edge (v0,v1) shared by triangles (v0,v1,v2) and (v0,v1,v3):
//
//   Cotangent weights in triangle 0 (v0,v1,v2):
//     A0         = area (Heron's formula from l01, l02, l12)
//     cot02      = cot(angle at v1) = (l01² + l12² - l02²) / (4·A0)
//     cot12      = cot(angle at v0) = (l01² + l02² - l12²) / (4·A0)
//   Cotangent weights in triangle 1 (v0,v1,v3):
//     A1         = area (Heron's formula from l01, l03, l13)
//     cot03      = cot(angle at v1) = (l01² + l13² - l03²) / (4·A1)
//     cot13      = cot(angle at v0) = (l01² + l03² - l13²) / (4·A1)
//
//   Vertex weights:
//     w0 =  cot02 + cot03    (v0: shared edge)
//     w1 =  cot12 + cot13    (v1: shared edge)
//     w2 = -(cot02 + cot12)  (v2: wing)
//     w3 = -(cot03 + cot13)  (v3: wing)
//
//   Rest curvature norm:
//     n_rest = |w0·X0 + w1·X1 + w2·X2 + w3·X3|  (≈ 0 for flat rest mesh)
// ============================================================================

void SimConstraints::build_bend(const ClothMesh& mesh,
                                const MeshTopology& topo,
                                float stiffness)
{
    bend_quads.clear();
    bend_w.clear();
    bend_n.clear();
    bend_stiffness.clear();

    for (const auto& et : topo.edges) {
        if (et.tri_b == -1) continue;  // boundary edge: no bend

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

        const Eigen::Vector3f& X0 = mesh.rest_pos[et.v0];
        const Eigen::Vector3f& X1 = mesh.rest_pos[et.v1];
        const Eigen::Vector3f& X2 = mesh.rest_pos[v2];
        const Eigen::Vector3f& X3 = mesh.rest_pos[v3];

        // Edge lengths in rest pose
        const float l01 = (X1 - X0).norm();
        const float l02 = (X2 - X0).norm();
        const float l03 = (X3 - X0).norm();
        const float l12 = (X2 - X1).norm();
        const float l13 = (X3 - X1).norm();

        // Triangle areas via Heron's formula
        auto heron = [](float a, float b, float c) -> float {
            const float s = 0.5f * (a + b + c);
            const float v = s * (s-a) * (s-b) * (s-c);
            return (v > 0.0f) ? std::sqrt(v) : 0.0f;
        };
        const float A0 = heron(l01, l02, l12);  // tri(v0,v1,v2)
        const float A1 = heron(l01, l03, l13);  // tri(v0,v1,v3)

        if (A0 < 1e-10f || A1 < 1e-10f) continue;  // degenerate triangle

        // Cotangent weights (law of cosines)
        // cot(angle_at_v1_in_tri0): opposite edge = l02
        const float cot02 = (l01*l01 + l12*l12 - l02*l02) / (4.0f * A0);
        // cot(angle_at_v0_in_tri0): opposite edge = l12
        const float cot12 = (l01*l01 + l02*l02 - l12*l12) / (4.0f * A0);
        // cot(angle_at_v1_in_tri1): opposite edge = l03
        const float cot03 = (l01*l01 + l13*l13 - l03*l03) / (4.0f * A1);
        // cot(angle_at_v0_in_tri1): opposite edge = l13
        const float cot13 = (l01*l01 + l03*l03 - l13*l13) / (4.0f * A1);

        const float w0 =   cot02 + cot03;
        const float w1 =   cot12 + cot13;
        const float w2 = -(cot02 + cot12);
        const float w3 = -(cot03 + cot13);

        // Rest curvature vector and its norm
        const Eigen::Vector3f e_rest = w0*X0 + w1*X1 + w2*X2 + w3*X3;
        const float n_rest = e_rest.norm();
        // n_rest ≈ 0 for flat cloth (discrete curvature is zero at rest)
        // When n_rest = 0, the projection drives current curvature → 0 (flatten)

        bend_quads.emplace_back(et.v0, et.v1, v2, v3);
        bend_w.push_back(w0);
        bend_w.push_back(w1);
        bend_w.push_back(w2);
        bend_w.push_back(w3);
        bend_n.push_back(n_rest);
        bend_stiffness.push_back(stiffness);
    }

    num_bend_cons = static_cast<int>(bend_quads.size());
    printf("  SimConstraints: %d bend constraints (cotangent, stiffness=%.4f)\n",
           num_bend_cons, stiffness);
}

// ============================================================================
// Jacobi diagonal precomputation
// ============================================================================

void SimConstraints::precompute_jacobi_diag(const ClothMesh& mesh, float dt)
{
    // diag_i = m_i + h² · Σ_{c∋i} (effective weight)²
    //
    // Triangle stretch  (Stiefel projection, F = Ds · G, G = Dm_inv):
    //   diag[v0] += h² · wA · ((g00+g10)² + (g01+g11)²)
    //   diag[v1] += h² · wA · (g00² + g01²)
    //   diag[v2] += h² · wA · (g10² + g11²)
    //   where wA = stretch_k[t] · rest_area[t]
    //
    // Cotangent bend:
    //   diag[v_i] += h² · bend_k · w_i²
    //   (w_i are the cotangent weights, possibly negative; squared → positive)

    const int N   = mesh.num_verts;
    const float h2 = dt * dt;

    std::vector<float> diag(N, 0.0f);

    // Mass
    for (int i = 0; i < N; ++i) diag[i] = mesh.mass[i];

    // Triangle stretch
    for (int t = 0; t < mesh.num_tris; ++t) {
        if (t >= (int)tri_stretch_k.size()) break;
        const int v0 = mesh.triangles[t](0);
        const int v1 = mesh.triangles[t](1);
        const int v2 = mesh.triangles[t](2);
        const float wA = tri_stretch_k[t] * mesh.rest_area[t];
        const float g00 = mesh.Dm_inv[t](0, 0), g10 = mesh.Dm_inv[t](1, 0);
        const float g01 = mesh.Dm_inv[t](0, 1), g11 = mesh.Dm_inv[t](1, 1);
        const float s0  = g00 + g10, s1 = g01 + g11;
        diag[v0] += h2 * wA * (s0*s0 + s1*s1);
        diag[v1] += h2 * wA * (g00*g00 + g01*g01);
        diag[v2] += h2 * wA * (g10*g10 + g11*g11);
    }

    // Cotangent bend: w_i² contribution per vertex
    for (int e = 0; e < num_bend_cons; ++e) {
        const float k = bend_stiffness[e];
        for (int i = 0; i < 4; ++i) {
            const float wi = bend_w[e*4 + i];
            diag[bend_quads[e](i)] += h2 * k * wi * wi;
        }
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

    if (!tri_stretch_k.empty())
        alloc_copy((void**)&d_tri_stretch_k,
                   tri_stretch_k.data(), num_tris * sizeof(float));

    // d_jacobi_diag is handled by precompute_jacobi_diag(), not here.

    if (num_bend_cons > 0) {
        std::vector<int> h_quads(num_bend_cons * 4);
        for (int i = 0; i < num_bend_cons; ++i) {
            h_quads[i*4+0] = bend_quads[i](0);
            h_quads[i*4+1] = bend_quads[i](1);
            h_quads[i*4+2] = bend_quads[i](2);
            h_quads[i*4+3] = bend_quads[i](3);
        }
        alloc_copy((void**)&d_bend_quads, h_quads.data(),
                   num_bend_cons * 4 * sizeof(int));
        alloc_copy((void**)&d_bend_w, bend_w.data(),
                   num_bend_cons * 4 * sizeof(float));
        alloc_copy((void**)&d_bend_n, bend_n.data(),
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
    auto sf = [](void* p) { if (p) cudaFree(p); };
    sf(d_tri_stretch_k); d_tri_stretch_k = nullptr;
    sf(d_jacobi_diag);   d_jacobi_diag   = nullptr;
    sf(d_bend_quads);    d_bend_quads     = nullptr;
    sf(d_bend_w);        d_bend_w         = nullptr;
    sf(d_bend_n);        d_bend_n         = nullptr;
    sf(d_bend_k);        d_bend_k         = nullptr;
#endif
}
