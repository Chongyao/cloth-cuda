#include "cloth_mesh.h"
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
        const float expected = 0.1f * total_area;
        const float rel_err  = std::abs(total_mass - expected) / (expected + 1e-12f);
        CHECK(rel_err < 1e-5f);
    }

    SECTION("Dm_inv * Dm == I");
    {
        for (int t = 0; t < mesh.num_tris; ++t) {
            const int i0 = mesh.triangles[t](0);
            const int i1 = mesh.triangles[t](1);
            const int i2 = mesh.triangles[t](2);

            Eigen::Matrix<float, 3, 2> edgeVec;
            edgeVec.col(0) = mesh.rest_pos[i1] - mesh.rest_pos[i0];
            edgeVec.col(1) = mesh.rest_pos[i2] - mesh.rest_pos[i0];

            Eigen::Vector3f p0 = edgeVec.col(0).normalized();
            Eigen::Vector3f ortho = edgeVec.col(1) - edgeVec.col(1).dot(p0) * p0;
            Eigen::Vector3f p1 = ortho.normalized();
            Eigen::Matrix<float, 3, 2> P;
            P.col(0) = p0;
            P.col(1) = p1;

            Eigen::Matrix2f Dm = P.transpose() * edgeVec;
            const float err = (mesh.Dm_inv[t] * Dm - Eigen::Matrix2f::Identity()).norm();
            CHECK(err < 1e-4f);
        }
    }

    SECTION("dF_dx maps rest pose to Dm_inv-consistent F");
    {
        for (int t = 0; t < std::min(mesh.num_tris, 10); ++t) {
            const int i0 = mesh.triangles[t](0);
            const int i1 = mesh.triangles[t](1);
            const int i2 = mesh.triangles[t](2);
            Eigen::Matrix<float, 9, 1> x;
            x.segment<3>(0) = mesh.rest_pos[i0];
            x.segment<3>(3) = mesh.rest_pos[i1];
            x.segment<3>(6) = mesh.rest_pos[i2];

            Eigen::Matrix<float, 6, 1> Fvec = mesh.dF_dx[t] * x;

            Eigen::Matrix<float, 3, 2> Ds;
            Ds.col(0) = mesh.rest_pos[i1] - mesh.rest_pos[i0];
            Ds.col(1) = mesh.rest_pos[i2] - mesh.rest_pos[i0];
            Eigen::Matrix<float, 3, 2> F = Ds * mesh.Dm_inv[t];
            Eigen::Matrix<float, 6, 1> expected;
            expected.segment<3>(0) = F.col(0);
            expected.segment<3>(3) = F.col(1);

            CHECK((Fvec - expected).norm() < 1e-4f);
        }
    }

    return test_summary();
}
