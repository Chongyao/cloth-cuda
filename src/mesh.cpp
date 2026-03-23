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
    safe_free(d_tri_stretch_k); d_tri_stretch_k = nullptr;
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

// ==== New Topology System ====

void ClothMesh::build_topology() {
    // Clear existing
    edges.clear();
    vert_to_tris.clear();
    tri_to_tris.clear();

    // Resize vertex-to-triangle adjacency
    vert_to_tris.resize(num_verts);
    for (int t = 0; t < num_tris; ++t) {
        vert_to_tris[triangles[t](0)].push_back(t);
        vert_to_tris[triangles[t](1)].push_back(t);
        vert_to_tris[triangles[t](2)].push_back(t);
    }

    // Build edge topology using hash map
    // Key: (v0, v1) with v0 < v1
    // Value: (tri_a, tri_b) where tri_b = -1 initially
    struct EdgeKey {
        int v0, v1;
        bool operator==(const EdgeKey& o) const { return v0 == o.v0 && v1 == o.v1; }
    };
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& e) const {
            return (static_cast<size_t>(e.v0) << 32) | static_cast<size_t>(e.v1);
        }
    };

    std::unordered_map<EdgeKey, std::pair<int, int>, EdgeKeyHash> edge_map;
    edge_map.reserve(num_tris * 3);

    // For each triangle, add its 3 edges
    for (int t = 0; t < num_tris; ++t) {
        int v[3] = {triangles[t](0), triangles[t](1), triangles[t](2)};
        for (int e = 0; e < 3; ++e) {
            int a = v[e], b = v[(e+1)%3];
            if (a > b) std::swap(a, b);
            EdgeKey key{a, b};
            auto it = edge_map.find(key);
            if (it == edge_map.end()) {
                edge_map[key] = {t, -1};  // First triangle
            } else {
                it->second.second = t;    // Second triangle
            }
        }
    }

    // Convert to EdgeTopo array
    edges.reserve(edge_map.size());
    for (auto& kv : edge_map) {
        EdgeTopo et;
        et.v0 = kv.first.v0;
        et.v1 = kv.first.v1;
        et.tri_a = kv.second.first;
        et.tri_b = kv.second.second;
        // Compute rest length
        Eigen::Vector3f p0 = rest_pos[et.v0];
        Eigen::Vector3f p1 = rest_pos[et.v1];
        et.rest_len = (p1 - p0).norm();
        edges.push_back(et);
    }

    // Build tri_to_tris: for each triangle, find adjacent triangles
    // tri_to_tris[t][i] = triangle sharing edge opposite to vertex i
    tri_to_tris.resize(num_tris, Eigen::Vector3i(-1, -1, -1));
    for (int e = 0; e < edges.size(); ++e) {
        const EdgeTopo& et = edges[e];
        if (et.tri_b == -1) continue;  // Boundary edge

        // Find which vertex is opposite to this edge in each triangle
        auto find_opposite = [&](int tri_idx, int v0, int v1) -> int {
            for (int i = 0; i < 3; ++i) {
                int v = triangles[tri_idx](i);
                if (v != v0 && v != v1) return i;  // Return local vertex index 0,1,2
            }
            return -1;
        };

        int opp_a = find_opposite(et.tri_a, et.v0, et.v1);
        int opp_b = find_opposite(et.tri_b, et.v0, et.v1);

        if (opp_a != -1) tri_to_tris[et.tri_a][opp_a] = et.tri_b;
        if (opp_b != -1) tri_to_tris[et.tri_b][opp_b] = et.tri_a;
    }

    printf("=== Topology Build ===\n");
    printf("  Vertices: %d\n", num_verts);
    printf("  Triangles: %d\n", num_tris);
    printf("  Total edges: %zu\n", edges.size());
    int boundary_edges = 0, inner_edges_count = 0;
    for (const auto& e : edges) {
        if (e.tri_b == -1) boundary_edges++;
        else inner_edges_count++;
    }
    printf("    - Boundary edges: %d\n", boundary_edges);
    printf("    - Inner edges (bendable): %d\n", inner_edges_count);
}

void ClothMesh::build_stretch_from_topo(float stiffness) {
    // Build stretch constraints from edge topology
    // Each edge becomes a stretch constraint
    stretch_constraints.clear();
    stretch_constraints.reserve(edges.size());

    for (const auto& et : edges) {
        // Store: (v0, v1, rest_len, stiffness)
        stretch_constraints.emplace_back(et.v0, et.v1, et.rest_len, stiffness);
    }
    num_stretch_cons = static_cast<int>(stretch_constraints.size());

    printf("  Stretch constraints from topo: %d\n", num_stretch_cons);
}

void ClothMesh::build_bend_from_topo(float stiffness) {
    // Build bend constraints from inner edges (edges with two adjacent triangles)
    // Each inner edge becomes a bend constraint
    inner_edges.clear();
    bend_rest_angles.clear();
    bend_stiffness.clear();

    for (const auto& et : edges) {
        if (et.tri_b == -1) continue;  // Skip boundary edges

        // Find opposite vertices in each triangle
        auto find_opposite = [&](int tri_idx, int v0, int v1) -> int {
            for (int i = 0; i < 3; ++i) {
                int v = triangles[tri_idx](i);
                if (v != v0 && v != v1) return v;
            }
            return -1;
        };

        int v2 = find_opposite(et.tri_a, et.v0, et.v1);
        int v3 = find_opposite(et.tri_b, et.v0, et.v1);

        if (v2 == -1 || v3 == -1) continue;

        // Compute rest dihedral angle using wing-vector convention
        Eigen::Vector3f p0 = rest_pos[et.v0];
        Eigen::Vector3f p1 = rest_pos[et.v1];
        Eigen::Vector3f p2 = rest_pos[v2];
        Eigen::Vector3f p3 = rest_pos[v3];

        Eigen::Vector3f edge = p1 - p0;
        float edge_len = edge.norm();
        if (edge_len < 1e-10f) continue;
        Eigen::Vector3f ax = edge / edge_len;

        // Wing vectors
        auto wing = [&](const Eigen::Vector3f& p) -> Eigen::Vector3f {
            float t = (p - p0).dot(ax);
            return (p - p0) - t * ax;
        };

        Eigen::Vector3f r2 = wing(p2);
        Eigen::Vector3f r3 = wing(p3);
        float r2_len = r2.norm(), r3_len = r3.norm();

        if (r2_len < 1e-10f || r3_len < 1e-10f) continue;

        Eigen::Vector3f r2h = r2 / r2_len;
        Eigen::Vector3f r3h = r3 / r3_len;

        float cos_t = std::clamp(r2h.dot(r3h), -1.0f, 1.0f);
        float sin_t = r2h.cross(r3h).dot(ax);
        float rest_angle = std::atan2(sin_t, cos_t);

        inner_edges.emplace_back(et.v0, et.v1, v2, v3);
        bend_rest_angles.push_back(rest_angle);
        bend_stiffness.push_back(stiffness);
    }
    num_bend_cons = static_cast<int>(inner_edges.size());
    num_inner_edges = num_bend_cons;

    printf("  Bend constraints from topo: %d\n", num_bend_cons);
}

void ClothMesh::build_tri_stretch(float stiffness) {
    // Build triangle-based stretch constraints (Stiefel projection)
    // Each triangle has a stiffness weight for its strain constraint
    tri_stretch_k.assign(num_tris, stiffness);
    printf("  Triangle stretch constraints: %d\n", num_tris);
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

    // Upload bend constraints
    if (!bend_rest_angles.empty()) {
        int E = num_inner_edges;
        // Flatten CPU inner_edges (Vector4i) to int array
        std::vector<int> h_quads(E * 4);
        for (int i = 0; i < E; ++i) {
            h_quads[i*4+0] = inner_edges[i](0);
            h_quads[i*4+1] = inner_edges[i](1);
            h_quads[i*4+2] = inner_edges[i](2);
            h_quads[i*4+3] = inner_edges[i](3);
        }
        alloc_and_copy((void**)&d_bend_quads, h_quads.data(), E * 4 * sizeof(int));
        alloc_and_copy((void**)&d_bend_rest, bend_rest_angles.data(), E * sizeof(float));
        std::vector<float> h_bend_k(E);
        for (int i = 0; i < E; ++i) h_bend_k[i] = bend_stiffness[i];
        alloc_and_copy((void**)&d_bend_k, h_bend_k.data(), E * sizeof(float));
    }

    // Upload triangle-based stretch stiffness (if defined)
    if (!tri_stretch_k.empty()) {
        alloc_and_copy((void**)&d_tri_stretch_k, tri_stretch_k.data(), T * sizeof(float));
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

        // Compute dihedral angle using the SAME wing-vector convention as
        // bend_project_kernel: angle between the perpendicular components of
        // v2 and v3 relative to the shared edge axis.
        // For a flat cloth this gives ~π (v2 and v3 on opposite sides).
        Eigen::Vector3f p0 = rest_pos[v0];
        Eigen::Vector3f p1 = rest_pos[v1];
        Eigen::Vector3f p2 = rest_pos[v2];
        Eigen::Vector3f p3 = rest_pos[v3];

        Eigen::Vector3f edge = p1 - p0;
        float edge_len = edge.norm();
        if (edge_len < 1e-10f) {
            bend_rest_angles[e] = static_cast<float>(M_PI);
            continue;
        }
        Eigen::Vector3f ax = edge / edge_len;

        // Wing vectors: perpendicular component of v2 and v3 from the edge
        auto wing = [&](const Eigen::Vector3f& p) {
            float t = (p - p0).dot(ax);
            return (p - p0) - t * ax;
        };
        Eigen::Vector3f r2 = wing(p2);
        Eigen::Vector3f r3 = wing(p3);
        float r2_len = r2.norm(), r3_len = r3.norm();

        if (r2_len < 1e-10f || r3_len < 1e-10f) {
            bend_rest_angles[e] = static_cast<float>(M_PI);
            continue;
        }
        Eigen::Vector3f r2h = r2 / r2_len;
        Eigen::Vector3f r3h = r3 / r3_len;

        float cos_t = std::clamp(r2h.dot(r3h), -1.0f, 1.0f);
        float sin_t = r2h.cross(r3h).dot(ax);
        bend_rest_angles[e] = std::atan2(sin_t, cos_t);
    }
    num_bend_cons = num_inner_edges;
}

void ClothMesh::precompute_jacobi_diag(float dt, float /*constraint_wt*/) {
    // Compute diagonal of system matrix A = M + h² * Σ w_i * A_i^T * A_i
    // For stretch constraint on edge (i,j):
    //   A_i^T * A_i contributes [ w, -w; -w, w ] to the 2x2 block
    // For bend constraint on quad (i,j,k,l): more complex (Phase 4)

    std::vector<float> diag(num_verts, 0.0f);

    // Mass contribution
    for (int i = 0; i < num_verts; ++i) {
        diag[i] = mass[i];
    }

    float h2 = dt * dt;

    // Triangle-based stretch (Stiefel projection)
    // With Ds = [x1-x0, x2-x0] and F = Ds * Dm_inv where G = Dm_inv:
    // Contribution to diagonal:
    //   diag[v0] += h² * wA * ((g00+g10)² + (g01+g11)²)  [sum of columns, squared]
    //   diag[v1] += h² * wA * (g00² + g01²)
    //   diag[v2] += h² * wA * (g10² + g11²)
    if (!tri_stretch_k.empty()) {
        for (int t = 0; t < num_tris; ++t) {
            int v0 = triangles[t](0);
            int v1 = triangles[t](1);
            int v2 = triangles[t](2);
            float wA = tri_stretch_k[t] * rest_area[t];
            float g00 = Dm_inv[t](0, 0);  // col 0, row 0
            float g10 = Dm_inv[t](1, 0);  // col 0, row 1
            float g01 = Dm_inv[t](0, 1);  // col 1, row 0
            float g11 = Dm_inv[t](1, 1);  // col 1, row 1
            float s0 = g00 + g10;
            float s1 = g01 + g11;
            diag[v0] += h2 * wA * (s0 * s0 + s1 * s1);
            diag[v1] += h2 * wA * (g00 * g00 + g01 * g01);
            diag[v2] += h2 * wA * (g10 * g10 + g11 * g11);
        }
    }

    // Bend: only v2 and v3 (wing vertices) contribute — v0/v1 (shared edge)
    // are projected to their current positions, so they must NOT be counted
    // here or they gain an artificial "pull toward current pos" term.
    for (int e = 0; e < num_inner_edges; ++e) {
        float w = bend_stiffness.empty() ? 0.0f : bend_stiffness[e];
        if (w == 0.0f) continue;
        diag[inner_edges[e](2)] += h2 * w;
        diag[inner_edges[e](3)] += h2 * w;
    }

#ifdef CUDA_MS_HAVE_CUDA
    if (d_jacobi_diag) cudaFree(d_jacobi_diag);
    cudaMalloc((void**)&d_jacobi_diag, num_verts * sizeof(float));
    cudaMemcpy(d_jacobi_diag, diag.data(), num_verts * sizeof(float), cudaMemcpyHostToDevice);
#endif
}
