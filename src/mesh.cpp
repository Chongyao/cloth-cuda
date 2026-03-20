#include "mesh.h"

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <algorithm>

#ifdef CUDA_MS_HAVE_CUDA
#  include <cuda_runtime.h>
#endif

// ---- Destructor ----

ClothMesh::~ClothMesh() {
    free_gpu();
}

void ClothMesh::free_gpu() {
#ifdef CUDA_MS_HAVE_CUDA
    auto safe_free = [](void* p) { if (p) cudaFree(p); };
    safe_free(d_pos);       d_pos       = nullptr;
    safe_free(d_vel);       d_vel       = nullptr;
    safe_free(d_tris);      d_tris      = nullptr;
    safe_free(d_Dm_inv);    d_Dm_inv    = nullptr;
    safe_free(d_rest_area); d_rest_area = nullptr;
    safe_free(d_mass);      d_mass      = nullptr;
#endif
}

// ---- OBJ loading ----

bool ClothMesh::load_obj(const std::string& path) {
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
                vids.push_back(v - 1);   // OBJ is 1-indexed
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

// ---- Precompute rest state ----

void ClothMesh::precompute_rest_state(float density) {
    Dm_inv.resize(num_tris);
    rest_area.resize(num_tris);
    mass.assign(num_verts, 0.0f);

    for (int t = 0; t < num_tris; ++t) {
        int i0 = triangles[t](0);
        int i1 = triangles[t](1);
        int i2 = triangles[t](2);

        const Eigen::Vector3f& X0 = rest_pos[i0];
        const Eigen::Vector3f& X1 = rest_pos[i1];
        const Eigen::Vector3f& X2 = rest_pos[i2];

        Eigen::Vector3f e1 = X1 - X0;
        Eigen::Vector3f e2 = X2 - X0;

        float e1_len = e1.norm();

        // Project e2 onto the local 2-D plane spanned by e1
        float proj  = e2.dot(e1) / e1_len;
        float perp2 = e2.squaredNorm() - proj * proj;
        float perp  = (perp2 > 0.0f) ? std::sqrt(perp2) : 0.0f;

        // Dm = [u1 | u2], u1 = (e1_len, 0), u2 = (proj, perp)
        Eigen::Matrix2f Dm;
        Dm.col(0) = Eigen::Vector2f(e1_len, 0.0f);
        Dm.col(1) = Eigen::Vector2f(proj, perp);

        float area   = 0.5f * std::abs(Dm.determinant());
        rest_area[t] = area;
        Dm_inv[t]    = Dm.inverse();

        float tri_mass = density * area;
        mass[i0] += tri_mass / 3.0f;
        mass[i1] += tri_mass / 3.0f;
        mass[i2] += tri_mass / 3.0f;
    }
}

// ---- GPU upload ----

void ClothMesh::upload_to_gpu() {
#ifdef CUDA_MS_HAVE_CUDA
    free_gpu();

    int N = num_verts;
    int T = num_tris;

    std::vector<float> h_pos(N * 3);
    for (int i = 0; i < N; ++i) {
        h_pos[i * 3 + 0] = rest_pos[i](0);
        h_pos[i * 3 + 1] = rest_pos[i](1);
        h_pos[i * 3 + 2] = rest_pos[i](2);
    }

    std::vector<int> h_tris(T * 3);
    for (int t = 0; t < T; ++t) {
        h_tris[t * 3 + 0] = triangles[t](0);
        h_tris[t * 3 + 1] = triangles[t](1);
        h_tris[t * 3 + 2] = triangles[t](2);
    }

    // Flatten Dm_inv: col-major 2×2 → [m00, m10, m01, m11]
    std::vector<float> h_Dm_inv(T * 4);
    for (int t = 0; t < T; ++t) {
        const Eigen::Matrix2f& M = Dm_inv[t];
        h_Dm_inv[t * 4 + 0] = M(0, 0);
        h_Dm_inv[t * 4 + 1] = M(1, 0);
        h_Dm_inv[t * 4 + 2] = M(0, 1);
        h_Dm_inv[t * 4 + 3] = M(1, 1);
    }

    auto alloc_and_copy = [](void** dptr, const void* hptr, size_t bytes) {
        cudaMalloc(dptr, bytes);
        cudaMemcpy(*dptr, hptr, bytes, cudaMemcpyHostToDevice);
    };

    alloc_and_copy((void**)&d_pos,       h_pos.data(),     N * 3 * sizeof(float));
    alloc_and_copy((void**)&d_tris,      h_tris.data(),    T * 3 * sizeof(int));
    alloc_and_copy((void**)&d_Dm_inv,    h_Dm_inv.data(),  T * 4 * sizeof(float));
    alloc_and_copy((void**)&d_rest_area, rest_area.data(), T     * sizeof(float));
    alloc_and_copy((void**)&d_mass,      mass.data(),      N     * sizeof(float));

    cudaMalloc((void**)&d_vel, N * 3 * sizeof(float));
    cudaMemset(d_vel, 0, N * 3 * sizeof(float));
#else
    fprintf(stderr, "upload_to_gpu: CUDA not compiled in — skipping.\n");
#endif
}

// ---- Diagnostics ----

void ClothMesh::print_stats() const {
    printf("=== ClothMesh stats ===\n");
    printf("  Vertices : %d\n", num_verts);
    printf("  Triangles: %d\n", num_tris);

    if (!rest_area.empty()) {
        float total_area = 0.0f;
        for (float a : rest_area) total_area += a;
        printf("  Total rest area : %.6f\n", total_area);
    }

    if (!mass.empty()) {
        float total_mass = 0.0f;
        for (float m : mass) total_mass += m;
        printf("  Total mesh mass : %.6f\n", total_mass);
    }

    int sample = std::min(num_tris, 3);
    for (int t = 0; t < sample; ++t) {
        printf("  tri[%d]: area=%.6f  Dm_inv=[[%.4f,%.4f],[%.4f,%.4f]]\n",
               t, rest_area[t],
               Dm_inv[t](0,0), Dm_inv[t](0,1),
               Dm_inv[t](1,0), Dm_inv[t](1,1));
    }

    printf("  GPU buffers: %s\n", (d_pos != nullptr) ? "allocated" : "not allocated");
}
