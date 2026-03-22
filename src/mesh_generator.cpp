#include "mesh_generator.h"

#include <Eigen/Dense>
#include <cassert>

void generate_square_cloth(int nrows, int ncols, float size, int type,
                           ClothMesh& mesh)
{
    assert(nrows >= 2 && ncols >= 2);

    mesh.rest_pos.clear();
    mesh.triangles.clear();

    const int ncells_r = nrows - 1;
    const int ncells_c = ncols - 1;

    // --- Vertices ---
    // type 3 (米字格) adds one center vertex per cell
    int num_centers = (type == 3) ? ncells_r * ncells_c : 0;
    mesh.rest_pos.reserve(nrows * ncols + num_centers);

    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            mesh.rest_pos.emplace_back(
                static_cast<float>(j) * size,
                0.0f,
                static_cast<float>(i) * size
            );
        }
    }

    // For type 3, append center vertices after grid vertices
    // center(i,j) index = nrows*ncols + i*ncells_c + j
    if (type == 3) {
        for (int i = 0; i < ncells_r; ++i) {
            for (int j = 0; j < ncells_c; ++j) {
                mesh.rest_pos.emplace_back(
                    (static_cast<float>(j) + 0.5f) * size,
                    0.0f,
                    (static_cast<float>(i) + 0.5f) * size
                );
            }
        }
    }
    mesh.num_verts = static_cast<int>(mesh.rest_pos.size());

    // --- Triangles ---
    int tris_per_cell = (type == 3) ? 4 : 2;
    mesh.triangles.reserve(ncells_r * ncells_c * tris_per_cell);

    for (int i = 0; i < ncells_r; ++i) {
        for (int j = 0; j < ncells_c; ++j) {
            int v0 = i       * ncols + j;
            int v1 = i       * ncols + (j + 1);
            int v2 = (i + 1) * ncols + j;
            int v3 = (i + 1) * ncols + (j + 1);

            if (type == 3) {
                // 米字格: center vertex splits quad into 4 triangles
                //   v0--v1
                //   |\ /|
                //   | c |   (isotropic, no diagonal bias)
                //   |/ \|
                //   v2--v3
                int c = nrows * ncols + i * ncells_c + j;
                mesh.triangles.emplace_back(v0, v1, c);
                mesh.triangles.emplace_back(v1, v3, c);
                mesh.triangles.emplace_back(v3, v2, c);
                mesh.triangles.emplace_back(v2, v0, c);
            } else {
                bool use_back_slash;
                if (type == 1) {
                    use_back_slash = ((i + j) % 2 == 0);
                } else if (type == 2) {
                    use_back_slash = false;
                } else {
                    use_back_slash = true;  // type 0 default
                }

                if (use_back_slash) {
                    mesh.triangles.emplace_back(v0, v1, v3);
                    mesh.triangles.emplace_back(v0, v3, v2);
                } else {
                    mesh.triangles.emplace_back(v0, v1, v2);
                    mesh.triangles.emplace_back(v1, v3, v2);
                }
            }
        }
    }
    mesh.num_tris = static_cast<int>(mesh.triangles.size());
}
