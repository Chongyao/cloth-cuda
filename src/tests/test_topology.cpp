#include "cloth_mesh.h"
#include "mesh_topology.h"
#include "mesh_generator.h"
#include "tests/test_framework.h"

// For an m×n vertex grid with type-0 triangulation:
//   triangles    = 2*(m-1)*(n-1)
//   total_edges  = (m-1)*n + m*(n-1) + (m-1)*(n-1)  [h + v + diag]
//   inner_edges  = total_edges - boundary_edges
//   boundary     = 2*(m-1) + 2*(n-1)  [perimeter]
//   So inner = (m-1)*n + m*(n-1) + (m-1)*(n-1) - 2*(m-1) - 2*(n-1)
//   For 3×3: inner = 2*3 + 3*2 + 2*2 - 2*2 - 2*2 = 6+6+4-4-4 = 8
//   For 4×4: inner = 3*4 + 4*3 + 3*3 - 2*3 - 2*3 = 12+12+9-6-6 = 21
//   For 10×10: inner = 9*10+10*9+9*9 - 2*9-2*9 = 90+90+81-18-18 = 225

int main() {
    SECTION("inner edge count by grid size (type 0)");
    {
        auto count_inner = [](int rows, int cols) {
            ClothMesh m;
            generate_square_cloth(rows, cols, 1.0f, 0, m);
            return MeshTopology::build(m).num_inner_edges();
        };
        CHECK_EQ(count_inner(3,  3),  8);
        CHECK_EQ(count_inner(4,  4),  21);
        CHECK_EQ(count_inner(10, 10), 225);
    }

    SECTION("inner edge count for 3x3, types 0/1/2");
    {
        for (int type : {0, 1, 2}) {
            ClothMesh m;
            generate_square_cloth(3, 3, 1.0f, type, m);
            CHECK_EQ(MeshTopology::build(m).num_inner_edges(), 8);
        }
    }

    SECTION("every edge has two distinct vertices");
    {
        ClothMesh m;
        generate_square_cloth(5, 5, 1.0f, 3, m);
        const MeshTopology topo = MeshTopology::build(m);
        for (const auto& e : topo.edges)
            CHECK(e.v0 != e.v1);
    }

    SECTION("vert_to_tris covers all triangles");
    {
        ClothMesh m;
        generate_square_cloth(4, 4, 1.0f, 0, m);
        const MeshTopology topo = MeshTopology::build(m);
        // Every triangle index must appear in vert_to_tris of each of its vertices
        for (int t = 0; t < m.num_tris; ++t) {
            for (int v = 0; v < 3; ++v) {
                const int vi = m.triangles[t](v);
                bool found = false;
                for (int ti : topo.vert_to_tris[vi])
                    if (ti == t) { found = true; break; }
                CHECK(found);
            }
        }
    }

    return test_summary();
}
