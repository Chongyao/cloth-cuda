#pragma once

// MeshTopology: CPU-only edge and adjacency analysis.
//
// Not included by any .cu file — no CUDACC guard needed.
// Build once from a ClothMesh, then pass (const ref) to SimConstraints::build_bend().

#include "cloth_mesh.h"
#include <Eigen/Dense>
#include <vector>

struct MeshTopology {
    struct EdgeTopo {
        int   v0, v1;      // vertex indices, always v0 < v1
        int   tri_a;       // first adjacent triangle
        int   tri_b;       // second adjacent triangle; -1 = boundary edge
        float rest_len;    // |rest_pos[v1] - rest_pos[v0]|
    };

    std::vector<EdgeTopo>             edges;        // all unique edges
    std::vector<std::vector<int>>     vert_to_tris; // [N] adjacency lists
    std::vector<Eigen::Vector3i>      tri_to_tris;  // [T] neighbours per triangle

    // Number of inner (non-boundary) edges
    int num_inner_edges() const;

    // Factory: build topology from mesh geometry.
    // mesh.rest_pos and mesh.triangles must be populated.
    static MeshTopology build(const ClothMesh& mesh);
};
