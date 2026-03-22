#include "mesh.h"

#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <utility>

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
    safe_free(d_pos);          d_pos          = nullptr;
    safe_free(d_vel);          d_vel          = nullptr;
    safe_free(d_prev_pos);     d_prev_pos     = nullptr;
    safe_free(d_tris);         d_tris         = nullptr;
    safe_free(d_Dm_inv);       d_Dm_inv       = nullptr;
    safe_free(d_rest_area);    d_rest_area    = nullptr;
    safe_free(d_mass);         d_mass         = nullptr;
    safe_free(d_inner_edges);  d_inner_edges  = nullptr;
    safe_free(d_stretch_edges); d_stretch_edges = nullptr;
    safe_free(d_stretch_rest);  d_stretch_rest  = nullptr;
    safe_free(d_stretch_k);     d_stretch_k     = nullptr;
    safe_free(d_jacobi_diag);   d_jacobi_diag   = nullptr;
    safe_free(d_bend_quads);    d_bend_quads    = nullptr;
    safe_free(d_bend_rest);     d_bend_rest     = nullptr;
    safe_free(d_bend_k);        d_bend_k        = nullptr;
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

// ---- Inner-edge topology ----

void ClothMesh::build_inner_edges() {
    // Hash for std::pair<int,int>
    struct EdgeHash {
        size_t operator()(const std::pair<int, int>& e) const {
            return std::hash<long long>()(
                (static_cast<long long>(e.first) << 32) | static_cast<unsigned>(e.second));
        }
    };

    // Map: canonical edge (a < b) -> {tri_a_idx, opposite_vertex_a, tri_b_idx, opposite_vertex_b}
    // tri_b_idx == -1 until a second triangle is found.
    struct EdgeRecord {
        int tri_a      = -1;
        int opp_a      = -1;
        int tri_b      = -1;
        int opp_b      = -1;
    };
    std::unordered_map<std::pair<int,int>, EdgeRecord, EdgeHash> edge_map;
    edge_map.reserve(num_tris * 3);

    auto add_edge = [&](int va, int vb, int tri_idx, int opp_vertex) {
        // Canonicalise so that first < second
        if (va > vb) std::swap(va, vb);
        auto key = std::make_pair(va, vb);
        auto it = edge_map.find(key);
        if (it == edge_map.end()) {
            EdgeRecord rec;
            rec.tri_a = tri_idx;
            rec.opp_a = opp_vertex;
            edge_map[key] = rec;
        } else {
            it->second.tri_b = tri_idx;
            it->second.opp_b = opp_vertex;
        }
    };

    for (int t = 0; t < num_tris; ++t) {
        int v0 = triangles[t](0);
        int v1 = triangles[t](1);
        int v2 = triangles[t](2);
        add_edge(v0, v1, t, v2);
        add_edge(v1, v2, t, v0);
        add_edge(v0, v2, t, v1);
    }

    inner_edges.clear();
    for (auto& kv : edge_map) {
        const EdgeRecord& rec = kv.second;
        if (rec.tri_b == -1) continue; // boundary edge

        // v0, v1 = shared edge; v2 = opposite in tri_a; v3 = opposite in tri_b
        inner_edges.emplace_back(kv.first.first, kv.first.second,
                                 rec.opp_a, rec.opp_b);
    }
    num_inner_edges = static_cast<int>(inner_edges.size());
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

    if (num_inner_edges > 0) {
        std::vector<int> h_ie(num_inner_edges * 4);
        for (int e = 0; e < num_inner_edges; ++e) {
            h_ie[e * 4 + 0] = inner_edges[e](0);
            h_ie[e * 4 + 1] = inner_edges[e](1);
            h_ie[e * 4 + 2] = inner_edges[e](2);
            h_ie[e * 4 + 3] = inner_edges[e](3);
        }
        alloc_and_copy((void**)&d_inner_edges, h_ie.data(),
                       num_inner_edges * 4 * sizeof(int));
    }

    cudaMalloc((void**)&d_vel, N * 3 * sizeof(float));
    cudaMemset(d_vel, 0, N * 3 * sizeof(float));

    // Allocate prev_pos for Chebyshev
    cudaMalloc((void**)&d_prev_pos, N * 3 * sizeof(float));
    cudaMemset(d_prev_pos, 0, N * 3 * sizeof(float));

    // Upload stretch constraints
    if (num_stretch_cons > 0) {
        std::vector<int> h_edges(num_stretch_cons * 2);
        std::vector<float> h_rest(num_stretch_cons);
        std::vector<float> h_k(num_stretch_cons);
        for (int i = 0; i < num_stretch_cons; ++i) {
            h_edges[i * 2 + 0] = static_cast<int>(stretch_constraints[i](0));
            h_edges[i * 2 + 1] = static_cast<int>(stretch_constraints[i](1));
            h_rest[i] = stretch_constraints[i](2);
            h_k[i] = stretch_constraints[i](3);
        }
        alloc_and_copy((void**)&d_stretch_edges, h_edges.data(), num_stretch_cons * 2 * sizeof(int));
        alloc_and_copy((void**)&d_stretch_rest, h_rest.data(), num_stretch_cons * sizeof(float));
        alloc_and_copy((void**)&d_stretch_k, h_k.data(), num_stretch_cons * sizeof(float));
    }

    // Upload bend constraints (Phase 4)
    if (!bend_rest_angles.empty()) {
        int E = num_inner_edges;
        alloc_and_copy((void**)&d_bend_quads, d_inner_edges, E * 4 * sizeof(int)); // reuse inner_edges
        cudaMalloc((void**)&d_bend_rest, E * sizeof(float));
        cudaMemcpy(d_bend_rest, bend_rest_angles.data(), E * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_bend_k, E * sizeof(float));
        std::vector<float> h_bend_k(E);
        for (int i = 0; i < E; ++i) h_bend_k[i] = bend_stiffness[i];
        cudaMemcpy(d_bend_k, h_bend_k.data(), E * sizeof(float), cudaMemcpyHostToDevice);
    }
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

    printf("  Inner edges: %d\n", num_inner_edges);
    printf("  Stretch constraints: %d\n", num_stretch_cons);
    printf("  GPU buffers: %s\n", (d_pos != nullptr) ? "allocated" : "not allocated");
}

// ---- PD Constraint Building ----

void ClothMesh::build_stretch_constraints(float stiffness) {
    // Build unique edge set from triangles
    struct Edge {
        int v0, v1;
        bool operator==(const Edge& other) const {
            return (v0 == other.v0 && v1 == other.v1) ||
                   (v0 == other.v1 && v1 == other.v0);
        }
    };
    struct EdgeHash {
        size_t operator()(const Edge& e) const {
            // Order-independent hash
            long long k = (static_cast<long long>(std::min(e.v0, e.v1)) << 32) |
                          static_cast<unsigned>(std::max(e.v0, e.v1));
            return std::hash<long long>()(k);
        }
    };

    std::unordered_map<Edge, float, EdgeHash> edge_map;
    edge_map.reserve(num_tris * 3);

    for (const auto& tri : triangles) {
        int v0 = tri(0), v1 = tri(1), v2 = tri(2);
        Edge edges[3] = {{v0, v1}, {v1, v2}, {v0, v2}};
        for (const auto& e : edges) {
            if (edge_map.find(e) == edge_map.end()) {
                // Compute rest length
                Eigen::Vector3f p0 = rest_pos[e.v0];
                Eigen::Vector3f p1 = rest_pos[e.v1];
                float len = (p1 - p0).norm();
                edge_map[e] = len;
            }
        }
    }

    stretch_constraints.clear();
    stretch_constraints.reserve(edge_map.size());
    for (const auto& kv : edge_map) {
        const Edge& e = kv.first;
        float rest_len = kv.second;
        stretch_constraints.emplace_back(e.v0, e.v1, rest_len, stiffness);
    }
    num_stretch_cons = static_cast<int>(stretch_constraints.size());
}

void ClothMesh::build_bend_constraints(float stiffness) {
    // Requires inner_edges to be built first
    if (inner_edges.empty()) {
        build_inner_edges();
    }

    bend_rest_angles.resize(num_inner_edges);
    bend_stiffness.assign(num_inner_edges, stiffness);

    for (int e = 0; e < num_inner_edges; ++e) {
        int v0 = inner_edges[e](0);
        int v1 = inner_edges[e](1);
        int v2 = inner_edges[e](2);  // opposite in tri A
        int v3 = inner_edges[e](3);  // opposite in tri B

        // Compute current dihedral angle
        Eigen::Vector3f p0 = rest_pos[v0];
        Eigen::Vector3f p1 = rest_pos[v1];
        Eigen::Vector3f p2 = rest_pos[v2];
        Eigen::Vector3f p3 = rest_pos[v3];

        Eigen::Vector3f n1 = (p1 - p0).cross(p2 - p0);
        Eigen::Vector3f n2 = (p3 - p0).cross(p1 - p0);

        if (n1.norm() > 1e-10f && n2.norm() > 1e-10f) {
            n1.normalize();
            n2.normalize();
            float cos_theta = std::clamp(n1.dot(n2), -1.0f, 1.0f);
            bend_rest_angles[e] = std::acos(cos_theta);
        } else {
            bend_rest_angles[e] = 0.0f;
        }
    }
}

void ClothMesh::precompute_jacobi_diag(float dt, float constraint_wt) {
    // Compute diagonal of system matrix A = M + h² * Σ w_i * A_i^T * A_i
    // For stretch constraint on edge (i,j):
    //   A_i^T * A_i contributes [ w, -w; -w, w ] to the 2x2 block
    // For bend constraint on quad (i,j,k,l): more complex (Phase 4)

    std::vector<float> diag(num_verts, 0.0f);

    // Mass contribution
    for (int i = 0; i < num_verts; ++i) {
        diag[i] = mass[i];
    }

    // Stretch constraint contribution
    float h2_w = dt * dt * constraint_wt;
    for (const auto& cons : stretch_constraints) {
        int v0 = static_cast<int>(cons(0));
        int v1 = static_cast<int>(cons(1));
        float w = cons(3);  // stiffness

        diag[v0] += h2_w * w * 2.0f;  // A^T*A = [1,-1;-1,1] for edge, diagonal is 2
        diag[v1] += h2_w * w * 2.0f;
    }

#ifdef CUDA_MS_HAVE_CUDA
    if (d_jacobi_diag) cudaFree(d_jacobi_diag);
    cudaMalloc((void**)&d_jacobi_diag, num_verts * sizeof(float));
    cudaMemcpy(d_jacobi_diag, diag.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
#endif
}
