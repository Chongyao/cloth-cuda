#include "mesh.h"
#include "mesh_generator.h"
#include "tests/test_framework.h"

#include <Eigen/Dense>
#include <cmath>

int main() {
    ClothMesh mesh;
    generate_square_cloth(10, 10, 0.1f, 0, mesh);
    mesh.precompute_rest_state(0.1f);

    SECTION("mass conservation");
    {
        float total_area = 0.0f, total_mass = 0.0f;
        for (float a : mesh.rest_area) total_area += a;
        for (float m : mesh.mass)      total_mass += m;
        float expected = 0.1f * total_area;
        float rel_err  = std::abs(total_mass - expected) / (expected + 1e-12f);
        CHECK(rel_err < 1e-5f);
    }

    SECTION("Dm_inv identity");
    {
        for (int t = 0; t < mesh.num_tris; ++t) {
            int i0 = mesh.triangles[t](0);
            int i1 = mesh.triangles[t](1);
            int i2 = mesh.triangles[t](2);

            Eigen::Vector3f e1 = mesh.rest_pos[i1] - mesh.rest_pos[i0];
            Eigen::Vector3f e2 = mesh.rest_pos[i2] - mesh.rest_pos[i0];

            float e1_len = e1.norm();
            float proj   = e2.dot(e1) / e1_len;
            float perp   = std::sqrt(std::max(0.0f, e2.squaredNorm() - proj * proj));

            Eigen::Matrix2f Dm;
            Dm.col(0) = Eigen::Vector2f(e1_len, 0.0f);
            Dm.col(1) = Eigen::Vector2f(proj, perp);

            float err = (mesh.Dm_inv[t] * Dm - Eigen::Matrix2f::Identity()).norm();
            CHECK(err < 1e-4f);
        }
    }

    return test_summary();
}
