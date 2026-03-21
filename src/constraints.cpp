#include "constraints.h"

#include <Eigen/Dense>
#include <cassert>
#include <cstdio>

// ---- Preset boundary conditions ----

void Constraints::pin_top_row(const ClothMesh& mesh, int ncols)
{
    assert(ncols <= mesh.num_verts);
    std::vector<int> indices;
    indices.reserve(ncols);
    for (int j = 0; j < ncols; ++j)
        indices.push_back(j);
    set_from_list(indices, mesh);
}

void Constraints::pin_corners(const ClothMesh& mesh, int ncols)
{
    assert(ncols <= mesh.num_verts);
    set_from_list({0, ncols - 1}, mesh);
}

void Constraints::set_from_list(const std::vector<int>& indices,
                                const ClothMesh& mesh)
{
    pinned_indices = indices;

    // Initialise target_positions from current rest positions
    target_positions.assign(mesh.num_verts * 3, 0.0f);
    for (int idx : pinned_indices) {
        assert(idx >= 0 && idx < mesh.num_verts);
        target_positions[idx * 3 + 0] = mesh.rest_pos[idx](0);
        target_positions[idx * 3 + 1] = mesh.rest_pos[idx](1);
        target_positions[idx * 3 + 2] = mesh.rest_pos[idx](2);
    }
}

// ---- Apply ----

void Constraints::apply_to_mesh(ClothMesh& mesh) const
{
    for (int idx : pinned_indices) {
        assert(idx >= 0 && idx < mesh.num_verts);
        mesh.rest_pos[idx](0) = target_positions[idx * 3 + 0];
        mesh.rest_pos[idx](1) = target_positions[idx * 3 + 1];
        mesh.rest_pos[idx](2) = target_positions[idx * 3 + 2];
    }
}

// ---- Utilities ----

void Constraints::clear()
{
    pinned_indices.clear();
    target_positions.clear();
}

void Constraints::print_stats() const
{
    printf("=== Constraints stats ===\n");
    printf("  Pinned vertices: %d\n", (int)pinned_indices.size());
    if (!pinned_indices.empty()) {
        printf("  Indices:");
        for (int idx : pinned_indices)
            printf(" %d", idx);
        printf("\n");
    }
}
