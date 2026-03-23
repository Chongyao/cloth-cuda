#include "cloth_mesh.h"
#include "mesh_generator.h"
#include "constraints.h"
#include "tests/test_framework.h"

int main() {
    ClothMesh mesh;
    generate_square_cloth(10, 10, 0.1f, 0, mesh);

    SECTION("pin_top_row");
    {
        Constraints cons;
        cons.pin_top_row(mesh, 10);
        CHECK_EQ((int)cons.pinned_indices.size(), 10);
        CHECK_EQ(cons.pinned_indices[0], 0);
        CHECK_EQ(cons.pinned_indices[9], 9);
    }

    SECTION("pin_corners");
    {
        Constraints cons;
        cons.pin_corners(mesh, 10);
        CHECK_EQ((int)cons.pinned_indices.size(), 2);
        CHECK_EQ(cons.pinned_indices[0], 0);
        CHECK_EQ(cons.pinned_indices[1], 9);
    }

    return test_summary();
}
