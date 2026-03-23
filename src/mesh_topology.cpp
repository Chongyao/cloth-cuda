#include "mesh_topology.h"

#include <unordered_map>
#include <utility>
#include <cstdio>

// ============================================================================
// MeshTopology::build
// ============================================================================

MeshTopology MeshTopology::build(const ClothMesh& mesh)
{
    MeshTopology topo;
    const int N = mesh.num_verts;
    const int T = mesh.num_tris;

    // ---- Vertex-to-triangle adjacency ----
    topo.vert_to_tris.resize(N);
    for (int t = 0; t < T; ++t) {
        topo.vert_to_tris[mesh.triangles[t](0)].push_back(t);
        topo.vert_to_tris[mesh.triangles[t](1)].push_back(t);
        topo.vert_to_tris[mesh.triangles[t](2)].push_back(t);
    }

    // ---- Edge map: canonical (v0 < v1) → (tri_a, tri_b) ----
    struct EdgeKey {
        int v0, v1;
        bool operator==(const EdgeKey& o) const { return v0 == o.v0 && v1 == o.v1; }
    };
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& e) const {
            return (static_cast<size_t>(e.v0) << 32) | static_cast<size_t>(e.v1);
        }
    };
    std::unordered_map<EdgeKey, std::pair<int,int>, EdgeKeyHash> edge_map;
    edge_map.reserve(T * 3);

    for (int t = 0; t < T; ++t) {
        const int v[3] = { mesh.triangles[t](0),
                           mesh.triangles[t](1),
                           mesh.triangles[t](2) };
        for (int e = 0; e < 3; ++e) {
            int a = v[e], b = v[(e + 1) % 3];
            if (a > b) std::swap(a, b);
            EdgeKey key{a, b};
            auto it = edge_map.find(key);
            if (it == edge_map.end())
                edge_map[key] = {t, -1};
            else
                it->second.second = t;
        }
    }

    // ---- Populate edges vector ----
    topo.edges.reserve(edge_map.size());
    for (const auto& kv : edge_map) {
        EdgeTopo et;
        et.v0      = kv.first.v0;
        et.v1      = kv.first.v1;
        et.tri_a   = kv.second.first;
        et.tri_b   = kv.second.second;
        et.rest_len = (mesh.rest_pos[et.v1] - mesh.rest_pos[et.v0]).norm();
        topo.edges.push_back(et);
    }

    // ---- Triangle-to-triangle adjacency ----
    topo.tri_to_tris.assign(T, Eigen::Vector3i(-1, -1, -1));
    for (const auto& et : topo.edges) {
        if (et.tri_b == -1) continue;

        // Find which local vertex index (0/1/2) is opposite to this edge in each triangle
        auto find_local_opp = [&](int tri, int v0, int v1) -> int {
            for (int i = 0; i < 3; ++i) {
                int v = mesh.triangles[tri](i);
                if (v != v0 && v != v1) return i;
            }
            return -1;
        };
        const int opp_a = find_local_opp(et.tri_a, et.v0, et.v1);
        const int opp_b = find_local_opp(et.tri_b, et.v0, et.v1);
        if (opp_a != -1) topo.tri_to_tris[et.tri_a][opp_a] = et.tri_b;
        if (opp_b != -1) topo.tri_to_tris[et.tri_b][opp_b] = et.tri_a;
    }

    // ---- Print summary ----
    int boundary = 0, inner = 0;
    for (const auto& et : topo.edges) {
        if (et.tri_b == -1) ++boundary;
        else                ++inner;
    }
    printf("=== MeshTopology::build ===\n");
    printf("  Vertices: %d  Triangles: %d\n", N, T);
    printf("  Total edges: %zu  (boundary: %d  inner: %d)\n",
           topo.edges.size(), boundary, inner);

    return topo;
}

// ============================================================================
// MeshTopology::num_inner_edges
// ============================================================================

int MeshTopology::num_inner_edges() const
{
    int count = 0;
    for (const auto& et : edges)
        if (et.tri_b != -1) ++count;
    return count;
}
