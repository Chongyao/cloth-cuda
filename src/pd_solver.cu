#include "pd_solver.h"
#include "mesh.h"
#include "constraints.h"
#include "utils/cuda_helper.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// ============================================================================
// CUDA Kernels
// ============================================================================

// Predict positions: y = x + h*v + h^2*g  (gravity is acceleration, not force)
__global__ void predict_kernel(
    const float* __restrict__ pos,
    const float* __restrict__ vel,
    const float* __restrict__ /*mass*/,
    float* __restrict__ predict,
    int N,
    float dt,
    float gravity)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float h2g = dt * dt * gravity;

    predict[idx * 3 + 0] = pos[idx * 3 + 0] + dt * vel[idx * 3 + 0];
    predict[idx * 3 + 1] = pos[idx * 3 + 1] + dt * vel[idx * 3 + 1] + h2g;
    predict[idx * 3 + 2] = pos[idx * 3 + 2] + dt * vel[idx * 3 + 2];
}

// Stretch constraint projection: project edge to rest length
__global__ void stretch_project_kernel(
    const float* __restrict__ pos,
    const int2* __restrict__ edges,
    const float* __restrict__ rest_len,
    const float* __restrict__ stiffness,
    float3* __restrict__ projections,
    int num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    int v0 = edges[e].x;
    int v1 = edges[e].y;

    float3 p0 = make_float3(pos[v0 * 3 + 0], pos[v0 * 3 + 1], pos[v0 * 3 + 2]);
    float3 p1 = make_float3(pos[v1 * 3 + 0], pos[v1 * 3 + 1], pos[v1 * 3 + 2]);

    float3 dir = make_float3(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z);
    float len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);

    float target_len = rest_len[e];

    if (len > 1e-10f) {
        // Normalize and scale to target length
        float scale = target_len / len;
        dir.x *= scale;
        dir.y *= scale;
        dir.z *= scale;
    } else {
        // Degenerate case: keep original positions
        dir.x = dir.y = dir.z = 0.0f;
    }

    // Center point
    float3 center = make_float3((p0.x + p1.x) * 0.5f, (p0.y + p1.y) * 0.5f, (p0.z + p1.z) * 0.5f);

    // Projections: p0_proj = center - dir/2, p1_proj = center + dir/2
    // Actually we want: p0_proj = center - 0.5 * dir_normalized * target_len
    // But dir is already scaled, so:
    float half_len = target_len * 0.5f;
    float3 dir_norm;
    if (len > 1e-10f) {
        dir_norm.x = (p1.x - p0.x) / len;
        dir_norm.y = (p1.y - p0.y) / len;
        dir_norm.z = (p1.z - p0.z) / len;
    } else {
        dir_norm.x = 1.0f; dir_norm.y = 0.0f; dir_norm.z = 0.0f;
    }

    projections[e * 2 + 0] = make_float3(
        center.x - dir_norm.x * half_len,
        center.y - dir_norm.y * half_len,
        center.z - dir_norm.z * half_len);
    projections[e * 2 + 1] = make_float3(
        center.x + dir_norm.x * half_len,
        center.y + dir_norm.y * half_len,
        center.z + dir_norm.z * half_len);
}

// Jacobi update: compute new positions from predictions and constraint projections
__global__ void jacobi_update_kernel(
    const float* __restrict__ predict,
    const float* __restrict__ mass,
    const float* __restrict__ jacobi_diag,
    const int2* __restrict__ stretch_edges,
    const float3* __restrict__ stretch_proj,
    const float* __restrict__ stretch_k,
    float* __restrict__ new_pos,
    int N,
    int num_stretch,
    float h2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Start with inertial term: M * y
    float m = mass[idx];
    float3 numerator = make_float3(
        m * predict[idx * 3 + 0],
        m * predict[idx * 3 + 1],
        m * predict[idx * 3 + 2]);

    // Add constraint contributions (accumulated via atomic operations in separate kernel)
    // For now, we'll use a simplified approach: iterate over constraints
    // This is less efficient but easier to implement

    // Actually, let's use the precomputed RHS approach
    // We'll have another kernel that computes RHS from projections

    float diag = jacobi_diag[idx];
    float inv_diag = (diag > 1e-10f) ? (1.0f / diag) : 0.0f;

    new_pos[idx * 3 + 0] = numerator.x * inv_diag;
    new_pos[idx * 3 + 1] = numerator.y * inv_diag;
    new_pos[idx * 3 + 2] = numerator.z * inv_diag;
}

// Add inertial term to RHS: rhs += M * predict
__global__ void add_inertial_rhs_kernel(
    float* __restrict__ rhs,
    const float* __restrict__ mass,
    const float* __restrict__ predict,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float m = mass[idx];
    atomicAdd(&rhs[idx * 3 + 0], m * predict[idx * 3 + 0]);
    atomicAdd(&rhs[idx * 3 + 1], m * predict[idx * 3 + 1]);
    atomicAdd(&rhs[idx * 3 + 2], m * predict[idx * 3 + 2]);
}

// Accumulate constraint contributions to RHS (using atomicAdd)
__global__ void accumulate_stretch_rhs_kernel(
    const int2* __restrict__ edges,
    const float3* __restrict__ projections,
    const float* __restrict__ stiffness,
    float* __restrict__ rhs,
    int num_edges,
    float h2)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;

    int v0 = edges[e].x;
    int v1 = edges[e].y;
    float w = stiffness[e] * h2;

    float3 p0 = projections[e * 2 + 0];
    float3 p1 = projections[e * 2 + 1];

    // Each constraint contributes: w * p to each vertex
    // Use atomicAdd for parallel accumulation
    atomicAdd(&rhs[v0 * 3 + 0], w * p0.x);
    atomicAdd(&rhs[v0 * 3 + 1], w * p0.y);
    atomicAdd(&rhs[v0 * 3 + 2], w * p0.z);

    atomicAdd(&rhs[v1 * 3 + 0], w * p1.x);
    atomicAdd(&rhs[v1 * 3 + 1], w * p1.y);
    atomicAdd(&rhs[v1 * 3 + 2], w * p1.z);
}

// Final Jacobi division: x_new = rhs / jacobi_diag
__global__ void jacobi_divide_kernel(
    float* __restrict__ new_pos,
    const float* __restrict__ rhs,
    const float* __restrict__ jacobi_diag,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float diag = jacobi_diag[idx];
    float inv_diag = (diag > 1e-10f) ? (1.0f / diag) : 0.0f;

    new_pos[idx * 3 + 0] = rhs[idx * 3 + 0] * inv_diag;
    new_pos[idx * 3 + 1] = rhs[idx * 3 + 1] * inv_diag;
    new_pos[idx * 3 + 2] = rhs[idx * 3 + 2] * inv_diag;
}

// Update velocity and position
__global__ void update_velocity_kernel(
    float* __restrict__ pos,
    float* __restrict__ vel,
    float* __restrict__ prev_pos,  // For Chebyshev warm start
    const float* __restrict__ new_pos,
    int N,
    float dt,
    float damping)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float3 old_p = make_float3(pos[idx * 3 + 0], pos[idx * 3 + 1], pos[idx * 3 + 2]);
    float3 new_p = make_float3(new_pos[idx * 3 + 0], new_pos[idx * 3 + 1], new_pos[idx * 3 + 2]);

    // Store previous position for next frame's Chebyshev
    if (prev_pos) {
        prev_pos[idx * 3 + 0] = old_p.x;
        prev_pos[idx * 3 + 1] = old_p.y;
        prev_pos[idx * 3 + 2] = old_p.z;
    }

    // Velocity update with damping
    float3 new_v = make_float3(
        (new_p.x - old_p.x) / dt * (1.0f - damping),
        (new_p.y - old_p.y) / dt * (1.0f - damping),
        (new_p.z - old_p.z) / dt * (1.0f - damping));

    // Update
    pos[idx * 3 + 0] = new_p.x;
    pos[idx * 3 + 1] = new_p.y;
    pos[idx * 3 + 2] = new_p.z;

    vel[idx * 3 + 0] = new_v.x;
    vel[idx * 3 + 1] = new_v.y;
    vel[idx * 3 + 2] = new_v.z;
}

// Apply pinned constraints
__global__ void apply_constraints_kernel(
    float* __restrict__ pos,
    float* __restrict__ vel,
    const int* __restrict__ pinned_indices,
    const float* __restrict__ target_pos,
    int num_pinned)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pinned) return;

    int idx = pinned_indices[i];
    pos[idx * 3 + 0] = target_pos[i * 3 + 0];
    pos[idx * 3 + 1] = target_pos[i * 3 + 1];
    pos[idx * 3 + 2] = target_pos[i * 3 + 2];

    vel[idx * 3 + 0] = 0.0f;
    vel[idx * 3 + 1] = 0.0f;
    vel[idx * 3 + 2] = 0.0f;
}

// Clear RHS buffer
__global__ void clear_rhs_kernel(float* rhs, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    rhs[idx * 3 + 0] = 0.0f;
    rhs[idx * 3 + 1] = 0.0f;
    rhs[idx * 3 + 2] = 0.0f;
}

// ============================================================================
// External wrapper for Constraints::apply_gpu
// ============================================================================

void launch_apply_constraints_kernel(
    float* d_pos, float* d_vel,
    const int* d_indices, const float* d_target,
    int num_pinned)
{
    if (num_pinned == 0) return;
    const int block_size = 256;
    const int num_blocks = (num_pinned + block_size - 1) / block_size;
    apply_constraints_kernel<<<num_blocks, block_size>>>(
        d_pos, d_vel, d_indices, d_target, num_pinned);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// PDSolver Implementation
// ============================================================================

PDSolver::PDSolver(const PDSolverConfig& config, ClothMesh& mesh)
    : config_(config)
    , num_verts_(mesh.num_verts)
    , num_stretch_cons_(mesh.num_stretch_cons)
    , num_bend_cons_(0)
{
    // Allocate GPU buffers
    allocate_buffers(num_verts_, num_stretch_cons_, 0);

    // Ensure mesh has uploaded stretch constraints
    if (!mesh.d_stretch_edges) {
        fprintf(stderr, "PDSolver ERROR: Mesh stretch constraints not uploaded to GPU\n");
        fprintf(stderr, "  Call mesh.build_stretch_constraints() and mesh.upload_to_gpu() first\n");
    }
    // Note: d_jacobi_diag is set by precompute_jacobi_diag(), which should be called before step()
}

PDSolver::~PDSolver()
{
    free_buffers();
}

void PDSolver::allocate_buffers(int N, int E_stretch, int E_bend)
{
    cudaMalloc((void**)&d_predict_, N * 3 * sizeof(float));
    cudaMalloc((void**)&d_rhs_, N * 3 * sizeof(float));
    cudaMalloc((void**)&d_prev_pos_, N * 3 * sizeof(float));
    cudaMemset(d_prev_pos_, 0, N * 3 * sizeof(float));
    cudaMalloc((void**)&d_new_pos_, N * 3 * sizeof(float));

    if (E_stretch > 0) {
        cudaMalloc((void**)&d_stretch_proj_, E_stretch * 2 * sizeof(float3));
    }
    if (E_bend > 0) {
        cudaMalloc((void**)&d_bend_proj_, E_bend * 4 * sizeof(float3));
    }
}

void PDSolver::free_buffers()
{
    if (d_predict_) { cudaFree(d_predict_); d_predict_ = nullptr; }
    if (d_rhs_) { cudaFree(d_rhs_); d_rhs_ = nullptr; }
    if (d_prev_pos_) { cudaFree(d_prev_pos_); d_prev_pos_ = nullptr; }
    if (d_new_pos_) { cudaFree(d_new_pos_); d_new_pos_ = nullptr; }
    if (d_stretch_proj_) { cudaFree(d_stretch_proj_); d_stretch_proj_ = nullptr; }
    if (d_bend_proj_) { cudaFree(d_bend_proj_); d_bend_proj_ = nullptr; }
}

void PDSolver::reset()
{
    omega_prev_ = 1.0f;
    omega_curr_ = 1.0f;
    iter_count_ = 0;
}

void PDSolver::step(ClothMesh& mesh, const Constraints& cons)
{
    const int block_size = 256;
    const int N = num_verts_;
    const int E_stretch = num_stretch_cons_;
    const float dt = config_.dt;
    const float h2 = dt * dt;

    // Save old position for velocity update later
    CUDA_CHECK(cudaMemcpy(d_prev_pos_, mesh.d_pos, N * 3 * sizeof(float), cudaMemcpyDeviceToDevice));

    // Step 1: Predict positions
    {
        int num_blocks = (N + block_size - 1) / block_size;
        predict_kernel<<<num_blocks, block_size>>>(
            mesh.d_pos, mesh.d_vel, mesh.d_mass, d_predict_,
            N, dt, config_.gravity);
        CUDA_CHECK(cudaGetLastError());
    }

    // Local-Global iterations (ping-pong between mesh.d_pos and d_new_pos_)
    float* d_pos_in = mesh.d_pos;
    float* d_pos_out = d_new_pos_;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // Local Step: Compute stretch projections
        if (E_stretch > 0) {
            int num_blocks = (E_stretch + block_size - 1) / block_size;
            stretch_project_kernel<<<num_blocks, block_size>>>(
                d_pos_in,
                reinterpret_cast<const int2*>(mesh.d_stretch_edges),
                mesh.d_stretch_rest,
                mesh.d_stretch_k,
                reinterpret_cast<float3*>(d_stretch_proj_),
                E_stretch);
            CUDA_CHECK(cudaGetLastError());
        }

        // Global Step: Clear and accumulate RHS
        {
            int num_blocks = (N + block_size - 1) / block_size;
            clear_rhs_kernel<<<num_blocks, block_size>>>(d_rhs_, N);
            CUDA_CHECK(cudaGetLastError());
        }

        // Start with inertial term: M * y
        // We need to add this - for now simplified as just copy predict
        // Actually proper formula: RHS = M*y + h^2 * sum(w_c * A_c^T * p_c)
        // M*y is the base, then we add constraint contributions

        // Add inertial contribution
        {
            int num_blocks = (N + block_size - 1) / block_size;
            // For now, simplify: numerator = M * predict
            // We'll modify jacobi_divide to use predict directly with mass
        }

        // Accumulate stretch constraint contributions
        if (E_stretch > 0) {
            int num_blocks = (E_stretch + block_size - 1) / block_size;
            accumulate_stretch_rhs_kernel<<<num_blocks, block_size>>>(
                reinterpret_cast<const int2*>(mesh.d_stretch_edges),
                reinterpret_cast<const float3*>(d_stretch_proj_),
                mesh.d_stretch_k,
                d_rhs_, E_stretch, h2);
            CUDA_CHECK(cudaGetLastError());
        }

        // Add inertial term: rhs += M * predict
        {
            int num_blocks = (N + block_size - 1) / block_size;
            add_inertial_rhs_kernel<<<num_blocks, block_size>>>(
                d_rhs_, mesh.d_mass, d_predict_, N);
            CUDA_CHECK(cudaGetLastError());
        }

        // Jacobi division: x_new = rhs / jacobi_diag
        {
            int num_blocks = (N + block_size - 1) / block_size;
            jacobi_divide_kernel<<<num_blocks, block_size>>>(
                d_pos_out, d_rhs_, mesh.d_jacobi_diag, N);
            CUDA_CHECK(cudaGetLastError());
        }

        // Apply constraints
        if (cons.num_pinned > 0) {
            int num_blocks = (cons.num_pinned + block_size - 1) / block_size;
            apply_constraints_kernel<<<num_blocks, block_size>>>(
                d_pos_out, mesh.d_vel,
                cons.d_pinned_indices, cons.d_target_pos, cons.num_pinned);
            CUDA_CHECK(cudaGetLastError());
        }

        // Chebyshev acceleration
        if (config_.use_chebyshev && iter > 0) {
            chebyshev_accelerate(d_pos_in, d_pos_out);
        }

        // Swap buffers
        std::swap(d_pos_in, d_pos_out);
    }

    // Ensure final result is in mesh.d_pos
    if (d_pos_in != mesh.d_pos) {
        CUDA_CHECK(cudaMemcpy(mesh.d_pos, d_pos_in, N * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    }

    // Step 4: Update velocity: v = (x_new - x_old) / dt
    // d_prev_pos_ has the old position saved at start of step
    {
        int num_blocks = (N + block_size - 1) / block_size;
        update_velocity_kernel<<<num_blocks, block_size>>>(
            d_prev_pos_, mesh.d_vel, nullptr,
            mesh.d_pos,
            N, dt, 0.0f);
        CUDA_CHECK(cudaGetLastError());
        // Copy new positions (already in mesh.d_pos, no copy needed)
    }

    // Re-apply constraints to velocity
    if (cons.num_pinned > 0) {
        int num_blocks = (cons.num_pinned + block_size - 1) / block_size;
        apply_constraints_kernel<<<num_blocks, block_size>>>(
            mesh.d_pos, mesh.d_vel,
            cons.d_pinned_indices, cons.d_target_pos, cons.num_pinned);
        CUDA_CHECK(cudaGetLastError());
    }

    iter_count_++;
}

void PDSolver::chebyshev_accelerate(float* d_pos, float* d_new_pos)
{
    // Chebyshev semi-iterative method
    if (iter_count_ == 0) {
        omega_prev_ = 1.0f;
        omega_curr_ = 1.0f;
    } else if (iter_count_ == 1) {
        float rho2 = config_.rho * config_.rho;
        omega_curr_ = 2.0f / (2.0f - rho2);
    } else {
        float rho2 = config_.rho * config_.rho;
        omega_curr_ = 4.0f / (4.0f - rho2 * omega_prev_);
    }

    // x_{k+1} = omega_k * (x_new - x_old) + x_old
    // where x_new is from Jacobi, x_old is previous iterate
    // For now, skip Chebyshev until we have proper buffer management

    omega_prev_ = omega_curr_;
}

// Simplified kernel implementations for compilation
void PDSolver::predict_positions(float* d_pos, float* d_vel)
{
    // Handled in step()
}

void PDSolver::local_step_stretch(const float* d_pos)
{
    // Handled in step()
}

void PDSolver::global_step_jacobi(float* d_pos, float* d_new_pos)
{
    // Handled in step()
}

void PDSolver::update_velocity(float* d_pos, float* d_vel, const float* d_new_pos)
{
    // Handled in step()
}
