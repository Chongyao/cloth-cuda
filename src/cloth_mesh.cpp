#include "cloth_mesh.h"

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <algorithm>

#ifdef CUDA_MS_HAVE_CUDA
#  include <cuda_runtime.h>
#endif

// ============================================================================
// GPU lifecycle
// ============================================================================

void ClothMesh::free_gpu()
{
#ifdef CUDA_MS_HAVE_CUDA
    auto safe_free = [](void* p) { if (p) cudaFree(p); };
    safe_free(d_pos);       d_pos       = nullptr;
    safe_free(d_vel);       d_vel       = nullptr;
    safe_free(d_mass);      d_mass      = nullptr;
    safe_free(d_tris);      d_tris      = nullptr;
    safe_free(d_Dm_inv);    d_Dm_inv    = nullptr;
    safe_free(d_rest_area); d_rest_area = nullptr;
#endif
}

void ClothMesh::upload_to_gpu()
{
#ifdef CUDA_MS_HAVE_CUDA
    free_gpu();

    const int N = num_verts;
    const int T = num_tris;

    auto alloc_copy = [](void** dst, const void* src, size_t bytes) {
        cudaMalloc(dst, bytes);
        cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice);
    };

    // Pack rest_pos as flat float array and upload as initial positions
    std::vector<float> h_pos(N * 3);
    for (int i = 0; i < N; ++i) {
        h_pos[i * 3 + 0] = rest_pos[i](0);
        h_pos[i * 3 + 1] = rest_pos[i](1);
        h_pos[i * 3 + 2] = rest_pos[i](2);
    }
    alloc_copy((void**)&d_pos, h_pos.data(), N * 3 * sizeof(float));

    // Velocities start at zero
    cudaMalloc((void**)&d_vel, N * 3 * sizeof(float));
    cudaMemset(d_vel, 0, N * 3 * sizeof(float));

    alloc_copy((void**)&d_mass, mass.data(), N * sizeof(float));

    // Triangle indices
    std::vector<int> h_tris(T * 3);
    for (int t = 0; t < T; ++t) {
        h_tris[t * 3 + 0] = triangles[t](0);
        h_tris[t * 3 + 1] = triangles[t](1);
        h_tris[t * 3 + 2] = triangles[t](2);
    }
    alloc_copy((void**)&d_tris, h_tris.data(), T * 3 * sizeof(int));

    // Dm_inv: col-major 2×2 → [m00, m10, m01, m11]
    std::vector<float> h_Dm_inv(T * 4);
    for (int t = 0; t < T; ++t) {
        const Eigen::Matrix2f& M = Dm_inv[t];
        h_Dm_inv[t * 4 + 0] = M(0, 0);
        h_Dm_inv[t * 4 + 1] = M(1, 0);
        h_Dm_inv[t * 4 + 2] = M(0, 1);
        h_Dm_inv[t * 4 + 3] = M(1, 1);
    }
    alloc_copy((void**)&d_Dm_inv, h_Dm_inv.data(), T * 4 * sizeof(float));

    alloc_copy((void**)&d_rest_area, rest_area.data(), T * sizeof(float));
#else
    fprintf(stderr, "ClothMesh::upload_to_gpu: CUDA not compiled in — skipping.\n");
#endif
}

// ============================================================================
// OBJ loading
// ============================================================================

bool ClothMesh::load_obj(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "ClothMesh::load_obj: cannot open '%s'\n", path.c_str());
        return false;
    }

    rest_pos.clear();
    triangles.clear();

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            rest_pos.emplace_back(x, y, z);
        } else if (tok == "f") {
            std::vector<int> vids;
            std::string ft;
            while (ss >> ft) {
                int v = 0;
                sscanf(ft.c_str(), "%d", &v);
                vids.push_back(v - 1);  // OBJ is 1-indexed
            }
            for (int i = 1; i + 1 < (int)vids.size(); ++i)
                triangles.emplace_back(vids[0], vids[i], vids[i + 1]);
        }
    }

    num_verts = (int)rest_pos.size();
    num_tris  = (int)triangles.size();

    if (num_verts == 0 || num_tris == 0) {
        fprintf(stderr, "ClothMesh::load_obj: empty mesh in '%s'\n", path.c_str());
        return false;
    }
    return true;
}

// ============================================================================
// Rest-state precomputation
// ============================================================================

void ClothMesh::precompute_rest_state(float density)
{
    Dm_inv.resize(num_tris);
    deltaUV.resize(num_tris);
    dF_dx.resize(num_tris);
    rest_area.resize(num_tris);
    mass.assign(num_verts, 0.0f);
    pos_cpu = rest_pos;
    vel_cpu.assign(num_verts, Eigen::Vector3f::Zero());

    for (int t = 0; t < num_tris; ++t) {
        const int i0 = triangles[t](0);
        const int i1 = triangles[t](1);
        const int i2 = triangles[t](2);

        const Eigen::Vector3f& X0 = rest_pos[i0];
        const Eigen::Vector3f& X1 = rest_pos[i1];
        const Eigen::Vector3f& X2 = rest_pos[i2];

        Eigen::Vector3f e1 = X1 - X0;
        Eigen::Vector3f e2 = X2 - X0;

        // DiffCloth-style rest frame construction.
        // edgeVec = [X1-X0, X2-X0]
        // P = orthonormal basis spanning the triangle plane via Gram-Schmidt
        // deltaUV = P^T * edgeVec
        // inv_deltaUV = deltaUV.inverse()
        Eigen::Matrix<float, 3, 2> edgeVec;
        edgeVec.col(0) = e1;
        edgeVec.col(1) = e2;

        Eigen::Matrix<float, 3, 2> P;
        P.col(0) = e1.normalized();
        Eigen::Vector3f ortho = e2 - e2.dot(P.col(0)) * P.col(0);
        if (ortho.norm() > 1e-10f) {
            P.col(1) = ortho.normalized();
        } else {
            // Degenerate triangle: arbitrary perpendicular basis
            if (std::abs(P(0,0)) <= std::abs(P(1,0)) && std::abs(P(0,0)) <= std::abs(P(2,0)))
                P.col(1) = Eigen::Vector3f(0.0f, -P(2,0), P(1,0)).normalized();
            else
                P.col(1) = Eigen::Vector3f(-P(1,0), P(0,0), 0.0f).normalized();
        }

        Eigen::Matrix2f Dm = P.transpose() * edgeVec;
        deltaUV[t] = Dm;
        Dm_inv[t]  = Dm.inverse();

        const float area = 0.5f * std::abs(Dm.determinant());
        rest_area[t] = area;

        // DiffCloth-style dF/dx = kron(inv_deltaUV^T * p, I3)
        // where p = [[-1, 1, 0],
        //            [-1, 0, 1]]
        Eigen::Matrix<float, 2, 3> p;
        p << -1.0f, 1.0f, 0.0f,
             -1.0f, 0.0f, 1.0f;
        Eigen::Matrix<float, 2, 3> D = Dm_inv[t].transpose() * p;

        Eigen::Matrix<float, 6, 9> deriv = Eigen::Matrix<float, 6, 9>::Zero();
        for (int row = 0; row < 2; ++row) {
            for (int v = 0; v < 3; ++v) {
                const float coeff = D(row, v);
                deriv.block<3,3>(row * 3, v * 3) = coeff * Eigen::Matrix3f::Identity();
            }
        }
        dF_dx[t] = deriv;

        const float tri_mass = density * area;
        mass[i0] += tri_mass / 3.0f;
        mass[i1] += tri_mass / 3.0f;
        mass[i2] += tri_mass / 3.0f;
    }
}

// ============================================================================
// Diagnostics
// ============================================================================

void ClothMesh::print_stats() const
{
    printf("=== ClothMesh stats ===\n");
    printf("  Vertices : %d\n", num_verts);
    printf("  Triangles: %d\n", num_tris);

    if (!rest_area.empty()) {
        float total_area = 0.0f, min_area = rest_area[0], max_area = rest_area[0];
        for (float a : rest_area) {
            total_area += a;
            min_area = std::min(min_area, a);
            max_area = std::max(max_area, a);
        }
        printf("  Total rest area : %.6f (min=%.6f, max=%.6f)\n", total_area, min_area, max_area);
    }
    if (!mass.empty()) {
        float total_mass = 0.0f, min_mass = mass[0], max_mass = mass[0];
        for (float m : mass) {
            total_mass += m;
            min_mass = std::min(min_mass, m);
            max_mass = std::max(max_mass, m);
        }
        printf("  Total mesh mass : %.6f (min=%.6f, max=%.6f)\n", total_mass, min_mass, max_mass);
    }

    const int sample = std::min(num_tris, 3);
    for (int t = 0; t < sample; ++t) {
        printf("  tri[%d]: area=%.6f  Dm_inv=[[%.4f,%.4f],[%.4f,%.4f]]\n",
               t, rest_area[t],
               Dm_inv[t](0,0), Dm_inv[t](0,1),
               Dm_inv[t](1,0), Dm_inv[t](1,1));
    }
    printf("  GPU buffers: %s\n", (d_pos != nullptr) ? "allocated" : "not allocated");
}
