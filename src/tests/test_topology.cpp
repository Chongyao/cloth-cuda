#include "mesh.h"
#include "mesh_generator.h"
#include "tests/test_framework.h"

int main() {
    SECTION("inner_edges count by grid size");
    {
        // For an m×n vertex grid: inner_edges = m*n + 2*(m-1)*(n-1) - 1 - 2*(m-1) - 2*(n-1)
        ClothMesh m3;
        generate_square_cloth(3, 3, 1.0f, 0, m3);
        m3.build_inner_edges();
        CHECK_EQ(m3.num_inner_edges, 8);

        ClothMesh m4;
        generate_square_cloth(4, 4, 1.0f, 0, m4);
        m4.build_inner_edges();
        CHECK_EQ(m4.num_inner_edges, 21);

        ClothMesh m10;
        generate_square_cloth(10, 10, 1.0f, 0, m10);
        m10.build_inner_edges();
        CHECK_EQ(m10.num_inner_edges, 225);
    }

    SECTION("inner_edges count for 3x3 across triangulation types");
    {
        for (int type : {0, 1, 2}) {
            ClothMesh m;
            generate_square_cloth(3, 3, 1.0f, type, m);
            m.build_inner_edges();
            CHECK_EQ(m.num_inner_edges, 8);
        }
    }

    return test_summary();
}
