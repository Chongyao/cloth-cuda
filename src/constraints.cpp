#include "constraints.h"
#include "pd_solver.h"   // declares launch_apply_constraints_kernel

#include <Eigen/Dense>
#include <cassert>
#include <cstdio>

#ifdef CUDA_MS_HAVE_CUDA
#include <cuda_runtime.h>
#endif

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

    target_positions.assign(mesh.num_verts * 3, 0.0f);
    for (int idx : pinned_indices) {
        assert(idx >= 0 && idx < mesh.num_verts);
        target_positions[idx * 3 + 0] = mesh.rest_pos[idx](0);
        target_positions[idx * 3 + 1] = mesh.rest_pos[idx](1);
        target_positions[idx * 3 + 2] = mesh.rest_pos[idx](2);
    }
}

void Constraints::reset_to_rest(ClothMesh& mesh) const
{
    // Writes to CPU rest_pos — use only for rest-state reset, not during sim.
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

// ---- GPU methods ----

void Constraints::upload_to_gpu()
{
#ifdef CUDA_MS_HAVE_CUDA
    free_gpu();

    num_pinned = static_cast<int>(pinned_indices.size());
    if (num_pinned == 0) return;

    cudaMalloc((void**)&d_pinned_indices, num_pinned * sizeof(int));
    cudaMemcpy(d_pinned_indices, pinned_indices.data(),
               num_pinned * sizeof(int), cudaMemcpyHostToDevice);

    // Extract target positions for pinned vertices only
    std::vector<float> pinned_targets(num_pinned * 3);
    for (int i = 0; i < num_pinned; ++i) {
        const int idx = pinned_indices[i];
        pinned_targets[i * 3 + 0] = target_positions[idx * 3 + 0];
        pinned_targets[i * 3 + 1] = target_positions[idx * 3 + 1];
        pinned_targets[i * 3 + 2] = target_positions[idx * 3 + 2];
    }
    cudaMalloc((void**)&d_target_pos, num_pinned * 3 * sizeof(float));
    cudaMemcpy(d_target_pos, pinned_targets.data(),
               num_pinned * 3 * sizeof(float), cudaMemcpyHostToDevice);
#endif
}

void Constraints::free_gpu()
{
#ifdef CUDA_MS_HAVE_CUDA
    if (d_pinned_indices) { cudaFree(d_pinned_indices); d_pinned_indices = nullptr; }
    if (d_target_pos)     { cudaFree(d_target_pos);     d_target_pos     = nullptr; }
    num_pinned = 0;
#endif
}

void Constraints::apply_gpu(float* d_pos, float* d_vel) const
{
#ifdef CUDA_MS_HAVE_CUDA
    if (num_pinned == 0 || d_pinned_indices == nullptr) return;
    launch_apply_constraints_kernel(d_pos, d_vel,
                                    d_pinned_indices, d_target_pos, num_pinned);
#endif
}
