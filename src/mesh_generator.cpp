#include "mesh_generator.h"

#include <Eigen/Dense>
#include <cassert>

void generate_square_cloth(int nrows, int ncols, float size, int type,
                           ClothMesh& mesh)
{
    assert(nrows >= 2 && ncols >= 2);

    mesh.rest_pos.clear();
    mesh.triangles.clear();

    // --- Vertices ---
    mesh.rest_pos.reserve(nrows * ncols);
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            // Layout: x = col * size, y = 0, z = row * size
            mesh.rest_pos.emplace_back(
                static_cast<float>(j) * size,
                0.0f,
                static_cast<float>(i) * size
            );
        }
    }
    mesh.num_verts = nrows * ncols;

    // --- Triangles ---
    mesh.triangles.reserve((nrows - 1) * (ncols - 1) * 2);
    for (int i = 0; i < nrows - 1; ++i) {
        for (int j = 0; j < ncols - 1; ++j) {
            int v0 = i       * ncols + j;
            int v1 = i       * ncols + (j + 1);
            int v2 = (i + 1) * ncols + j;
            int v3 = (i + 1) * ncols + (j + 1);

            bool use_back_slash; // true = \ diagonal (v0-v3), false = / diagonal (v1-v2)
            if (type == 1) {
                use_back_slash = ((i + j) % 2 == 0);
            } else if (type == 2) {
                use_back_slash = false;
            } else {
                // type 0 (default)
                use_back_slash = true;
            }

            if (use_back_slash) {
                // \ split: shared edge v0-v3
                mesh.triangles.emplace_back(v0, v1, v3);
                mesh.triangles.emplace_back(v0, v3, v2);
            } else {
                // / split: shared edge v1-v2
                mesh.triangles.emplace_back(v0, v1, v2);
                mesh.triangles.emplace_back(v1, v3, v2);
            }
        }
    }
    mesh.num_tris = static_cast<int>(mesh.triangles.size());
}
