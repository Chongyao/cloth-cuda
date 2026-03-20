#include "mesh.h"
#include "utils/cuda_helper.h"

#include <Eigen/Dense>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <string>

#ifdef CUDA_MS_HAVE_CUDA
#  include <cuda_runtime.h>
#endif

// Generate a simple unit-square test OBJ (4 vertices, 2 triangles)
static void write_test_obj(const char* path) {
    FILE* f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot write test OBJ to %s\n", path); return; }
    fprintf(f,
        "# Unit square: 4 vertices, 2 triangles\n"
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 1.0 1.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "f 1 2 3\n"
        "f 1 3 4\n"
    );
    fclose(f);
}

int main(int argc, char* argv[]) {
    std::string obj_path;
    if (argc >= 2) {
        obj_path = argv[1];
    } else {
        obj_path = "/tmp/cuda_ms_test.obj";
        write_test_obj(obj_path.c_str());
        printf("No OBJ argument given — using generated test mesh: %s\n\n",
               obj_path.c_str());
    }

    // ---- Device info ----
    printf("=== CUDA Devices ===\n");
    print_device_info();
    printf("\n");

    // ---- Load mesh ----
    ClothMesh mesh;
    if (!mesh.load_obj(obj_path)) {
        fprintf(stderr, "Failed to load OBJ.\n");
        return EXIT_FAILURE;
    }

    // ---- Precompute ----
    const float density = 0.1f;
    mesh.precompute_rest_state(density);
    mesh.print_stats();

    // ---- Verify: total_mass ≈ total_area × density ----
    {
        float total_area = 0.0f, total_mass = 0.0f;
        for (float a : mesh.rest_area) total_area += a;
        for (float m : mesh.mass)      total_mass += m;
        float expected = density * total_area;
        float rel_err  = std::abs(total_mass - expected) / (expected + 1e-12f);
        printf("\nMass verification: got=%.6f  expected=%.6f  rel_err=%.2e  %s\n",
               total_mass, expected, rel_err,
               rel_err < 1e-5f ? "PASS" : "FAIL");
    }

    // ---- Verify: Dm_inv * Dm ≈ I ----
    {
        int bad = 0;
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
            if (err > 1e-4f) {
                printf("  tri[%d] Dm_inv check FAIL: residual=%.2e\n", t, err);
                ++bad;
            }
        }
        printf("Dm_inv identity check: %d / %d triangles PASS\n",
               mesh.num_tris - bad, mesh.num_tris);
    }

    // ---- GPU upload ----
    if (cuda_device_available()) {
        printf("\nUploading to GPU...\n");
        mesh.upload_to_gpu();
        printf("GPU upload complete.\n");

#ifdef CUDA_MS_HAVE_CUDA
        // Spot-check: read back first vertex position
        float h_pos[3];
        cudaMemcpy(h_pos, mesh.d_pos, 3 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("d_pos[0] = (%.4f, %.4f, %.4f)  expected (%.4f, %.4f, %.4f)\n",
               h_pos[0], h_pos[1], h_pos[2],
               mesh.rest_pos[0](0), mesh.rest_pos[0](1), mesh.rest_pos[0](2));
#endif
    } else {
        printf("\nNo CUDA device — skipping GPU upload.\n");
    }

    return EXIT_SUCCESS;
}
