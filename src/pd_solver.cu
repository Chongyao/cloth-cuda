#include "pd_solver.h"
#include "cloth_mesh.h"
#include "sim_constraints.h"
#include "constraints.h"
#include "utils/cuda_helper.h"
#ifndef __CUDACC__
#include "stretch_reference.h"
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void predict_kernel(
    const float* __restrict__ pos,
    const float* __restrict__ vel,
    float*       __restrict__ predict,
    int N, float dt, float gravity)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float h2g = dt * dt * gravity;
    predict[idx*3+0] = pos[idx*3+0] + dt * vel[idx*3+0];
    predict[idx*3+1] = pos[idx*3+1] + dt * vel[idx*3+1] + h2g;
    predict[idx*3+2] = pos[idx*3+2] + dt * vel[idx*3+2];
}

// ============================================================================
// Stretch: Stiefel manifold projection
//
// Following DiffCloth (Triangle.cpp::projectToManifold):
// 1. F = Ds * Dm_inv  (deformation gradient, 3x2)
// 2. Gram-Schmidt on F columns to get orthonormal basis U = [u1, u2] (3x2)
//    u1 = F0 / |F0|
//    u2 = (F1 - (F1·u1)*u1) / |F1 - (F1·u1)*u1|
// 3. F_2D = U^T * F  (project to 2D, 2x2 matrix)
// 4. SVD of F_2D:  F_2D = V * Σ * W^T  (note: DiffCloth uses U,V but I'll use V,W)
// 5. R_2D = V * W^T  (closest rotation in 2D)
// 6. R = U * R_2D  (map back to 3D, 3x2 matrix with orthonormal columns)
//
// Output: proj[t*6 + 0..2] = R[:,0],  proj[t*6 + 3..5] = R[:,1]
// ============================================================================

__global__ void tri_stretch_project_kernel(
    const float* __restrict__ pos,     // [N*3]
    const int*   __restrict__ tris,    // [T*3]
    const float* __restrict__ Dm_inv,  // [T*4] col-major 2x2
    float*       __restrict__ proj,    // [T*6] output R (3x2)
    int num_tris)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    const int v0 = tris[t*3+0], v1 = tris[t*3+1], v2 = tris[t*3+2];
    const float3 p0 = make_float3(pos[v0*3+0], pos[v0*3+1], pos[v0*3+2]);
    const float3 p1 = make_float3(pos[v1*3+0], pos[v1*3+1], pos[v1*3+2]);
    const float3 p2 = make_float3(pos[v2*3+0], pos[v2*3+1], pos[v2*3+2]);

    // Ds columns: [p1-p0, p2-p0]
    const float3 ds0 = make_float3(p1.x-p0.x, p1.y-p0.y, p1.z-p0.z);
    const float3 ds1 = make_float3(p2.x-p0.x, p2.y-p0.y, p2.z-p0.z);

    // Dm_inv (col-major): [m00, m10, m01, m11]
    const float m00 = Dm_inv[t*4+0], m10 = Dm_inv[t*4+1];
    const float m01 = Dm_inv[t*4+2], m11 = Dm_inv[t*4+3];

    // F = Ds * Dm_inv (3x2 matrix, columns are F0, F1)
    const float3 F0 = make_float3(ds0.x*m00 + ds1.x*m10,
                                   ds0.y*m00 + ds1.y*m10,
                                   ds0.z*m00 + ds1.z*m10);
    const float3 F1 = make_float3(ds0.x*m01 + ds1.x*m11,
                                   ds0.y*m01 + ds1.y*m11,
                                   ds0.z*m01 + ds1.z*m11);

    // Step 2: Gram-Schmidt orthonormalization of F's columns
    // u1 = F0 / |F0|
    const float len_F0 = sqrtf(F0.x*F0.x + F0.y*F0.y + F0.z*F0.z);
    float3 u1;
    if (len_F0 > 1e-10f) {
        u1 = make_float3(F0.x/len_F0, F0.y/len_F0, F0.z/len_F0);
    } else {
        u1 = make_float3(1.0f, 0.0f, 0.0f);
    }

    // u2_perp = F1 - (F1·u1)*u1
    const float F1_dot_u1 = F1.x*u1.x + F1.y*u1.y + F1.z*u1.z;
    const float3 u2_perp = make_float3(F1.x - F1_dot_u1*u1.x,
                                        F1.y - F1_dot_u1*u1.y,
                                        F1.z - F1_dot_u1*u1.z);
    const float len_u2_perp = sqrtf(u2_perp.x*u2_perp.x + u2_perp.y*u2_perp.y + u2_perp.z*u2_perp.z);
    float3 u2;
    if (len_u2_perp > 1e-10f) {
        u2 = make_float3(u2_perp.x/len_u2_perp, u2_perp.y/len_u2_perp, u2_perp.z/len_u2_perp);
    } else {
        // Degenerate: F1 is parallel to F0, pick arbitrary perpendicular
        if (fabsf(u1.x) > 0.9f) {
            u2 = make_float3(-u1.y, u1.x, 0.0f);
        } else {
            u2 = make_float3(0.0f, -u1.z, u1.y);
        }
        const float len_u2 = sqrtf(u2.x*u2.x + u2.y*u2.y + u2.z*u2.z);
        u2 = make_float3(u2.x/len_u2, u2.y/len_u2, u2.z/len_u2);
    }

    // Step 3: F_2D = U^T * F (project to 2D space spanned by u1, u2)
    // F_2D is 2x2: [F_2D00, F_2D01; F_2D10, F_2D11]
    // First row: u1·F0, u1·F1
    // Second row: u2·F0, u2·F1
    const float F2d_00 = u1.x*F0.x + u1.y*F0.y + u1.z*F0.z;  // u1·F0
    const float F2d_01 = u1.x*F1.x + u1.y*F1.y + u1.z*F1.z;  // u1·F1
    const float F2d_10 = u2.x*F0.x + u2.y*F0.y + u2.z*F0.z;  // u2·F0
    const float F2d_11 = u2.x*F1.x + u2.y*F1.y + u2.z*F1.z;  // u2·F1

    // Step 4: SVD of 2x2 matrix F_2D
    // A = F_2D^T * F_2D
    const float A00 = F2d_00*F2d_00 + F2d_10*F2d_10;
    const float A01 = F2d_00*F2d_01 + F2d_10*F2d_11;
    const float A11 = F2d_01*F2d_01 + F2d_11*F2d_11;

    const float tr = (A00 + A11) * 0.5f;
    const float disc = sqrtf(fmaxf(0.0f, (A00 - A11)*(A00 - A11)*0.25f + A01*A01));
    const float lambda1 = tr + disc;
    const float lambda2 = tr - disc;

    // Eigenvectors of A (right singular vectors V of F_2D)
    float v1x, v1y, v2x, v2y;
    if (disc < 1e-10f || fabsf(A01) < 1e-10f) {
        // Matrix is effectively diagonal (or isotropic)
        // Eigenvectors are the standard basis
        v1x = 1.0f; v1y = 0.0f;
        v2x = 0.0f; v2y = 1.0f;
    } else {
        // v1 = [A01, lambda1 - A00] / norm
        const float tmp = lambda1 - A00;
        const float len = sqrtf(A01*A01 + tmp*tmp);
        if (len > 1e-10f) {
            v1x = A01 / len;
            v1y = tmp / len;
        } else {
            v1x = 1.0f; v1y = 0.0f;
        }
        // v2 perpendicular to v1
        v2x = -v1y;
        v2y = v1x;
    }

    // Step 5: R_2D = V * W^T where W are left singular vectors
    // Left singular vectors: W = F_2D * V * Sigma^{-1}
    const float sigma1 = sqrtf(fmaxf(0.0f, lambda1));
    const float sigma2 = sqrtf(fmaxf(0.0f, lambda2));
    const float EPS = 1e-10f;

    float w1x, w1y, w2x, w2y;
    if (sigma1 > EPS) {
        // w1 = F_2D * v1 / sigma1
        w1x = (F2d_00*v1x + F2d_01*v1y) / sigma1;
        w1y = (F2d_10*v1x + F2d_11*v1y) / sigma1;
    } else {
        w1x = 1.0f; w1y = 0.0f;
    }
    if (sigma2 > EPS) {
        // w2 = F_2D * v2 / sigma2
        w2x = (F2d_00*v2x + F2d_01*v2y) / sigma2;
        w2y = (F2d_10*v2x + F2d_11*v2y) / sigma2;
    } else {
        // Gram-Schmidt orthogonalize against w1
        const float Fv2x = F2d_00*v2x + F2d_01*v2y;
        const float Fv2y = F2d_10*v2x + F2d_11*v2y;
        const float dot = w1x*Fv2x + w1y*Fv2y;
        const float wx = Fv2x - dot*w1x;
        const float wy = Fv2y - dot*w1y;
        const float len = sqrtf(wx*wx + wy*wy);
        if (len > EPS) {
            w2x = wx / len; w2y = wy / len;
        } else {
            w2x = -w1y; w2y = w1x;
        }
    }

    // R_2D = V * W^T
    // V = [v1x, v2x; v1y, v2y], W = [w1x, w2x; w1y, w2y]
    // R_2D[i,j] = V[i,0]*W[j,0] + V[i,1]*W[j,1] = V[i,:] · W[j,:]
    // R_2D00 = v1x*w1x + v2x*w2x
    // R_2D01 = v1x*w1y + v2x*w2y
    // R_2D10 = v1y*w1x + v2y*w2x
    // R_2D11 = v1y*w1y + v2y*w2y
    const float R2d_00 = v1x*w1x + v2x*w2x;
    const float R2d_01 = v1x*w1y + v2x*w2y;
    const float R2d_10 = v1y*w1x + v2y*w2x;
    const float R2d_11 = v1y*w1y + v2y*w2y;

    // Step 6: R = U * R_2D (3x2 matrix)
    // R[:,0] = u1*R2d_00 + u2*R2d_10
    // R[:,1] = u1*R2d_01 + u2*R2d_11
    const float3 R0 = make_float3(u1.x*R2d_00 + u2.x*R2d_10,
                                   u1.y*R2d_00 + u2.y*R2d_10,
                                   u1.z*R2d_00 + u2.z*R2d_10);
    const float3 R1 = make_float3(u1.x*R2d_01 + u2.x*R2d_11,
                                   u1.y*R2d_01 + u2.y*R2d_11,
                                   u1.z*R2d_01 + u2.z*R2d_11);

    proj[t*6+0] = R0.x; proj[t*6+1] = R0.y; proj[t*6+2] = R0.z;
    proj[t*6+3] = R1.x; proj[t*6+4] = R1.y; proj[t*6+5] = R1.z;
}

// Accumulate triangle stretch RHS contributions.
// Following DiffCloth Triangle::stretchingForce for QUADRATIC energy:
// Energy = 0.5 * k * area * ||F - R||^2
// where F = Ds * G, G = Dm_inv, R is the projection (3x2 rotation)
//
// The constraint residual is (F - R), and we need:
//   rhs += h² * w * (∂F/∂x)^T * (F - projection)
//
// Since ∂F/∂x involves Dm_inv, the contribution is:
//   v0: -w * G^T * (F - R) * [1; 1]  (negative sum)
//   v1:  w * G^T * (F - R) * [1; 0]
//   v2:  w * G^T * (F - R) * [0; 1]
//
// But actually we use the simpler form from the original code:
//   rhs contribution = w * G^T * R  (since F cancels in the PD formulation)
//
// Wait, let me re-derive:
// For PD, local step gives projection p (the R we computed).
// Global step minimizes: ||x - y||²_M + h² * Σ w * ||A x - p||²
// The RHS for each constraint is: h² * w * A^T * p
//
// For triangle stretch, A is the gradient of F w.r.t. x.
// F = Ds * G where Ds = [x1-x0, x2-x0], G = Dm_inv
// So A^T * p = G^T * projected_residual
//
// Actually the standard PD formulation uses:
//   rhs_i += h² * w * (∂F/∂x_i)^T : R
// where : is Frobenius inner product.
//
// For vertex contributions:
//   v0: ∂F/∂x0 = -[g00, g01; g10, g11] = -G^T (as 3x2)
//   v1: ∂F/∂x1 = [g00, g01; 0, 0] ... no wait
//
// Let me be more careful. F = [F0, F1] where:
//   F0 = (x1-x0)*g00 + (x2-x0)*g10
//   F1 = (x1-x0)*g01 + (x2-x0)*g11
//
// ∂F0/∂x0 = -(g00 + g10)
// ∂F0/∂x1 = g00
// ∂F0/∂x2 = g10
// ∂F1/∂x0 = -(g01 + g11)
// ∂F1/∂x1 = g01
// ∂F1/∂x2 = g11
//
// The Frobenius inner product F:R = F0·R0 + F1·R1
// So contribution to rhs[v] is h² * w * (∂F0/∂x[v] * R0 + ∂F1/∂x[v] * R1)
__global__ void accumulate_tri_stretch_rhs_kernel(
    const int*   __restrict__ tris,
    const float* __restrict__ Dm_inv,
    const float* __restrict__ rest_area,
    const float* __restrict__ stretch_k,
    const float* __restrict__ proj,      // [T*6] = R (3x2 projection)
    float*       __restrict__ rhs,
    int num_tris, float h2)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    const int v0 = tris[t*3+0], v1 = tris[t*3+1], v2 = tris[t*3+2];
    const float3 R0 = make_float3(proj[t*6+0], proj[t*6+1], proj[t*6+2]);
    const float3 R1 = make_float3(proj[t*6+3], proj[t*6+4], proj[t*6+5]);

    // G = Dm_inv (col-major): [g00, g10, g01, g11]
    const float g00 = Dm_inv[t*4+0], g10 = Dm_inv[t*4+1];
    const float g01 = Dm_inv[t*4+2], g11 = Dm_inv[t*4+3];

    const float wA = stretch_k[t] * rest_area[t] * h2;

    // Contribution to each vertex from R0 and R1
    // v0: -(g00+g10)*R0 - (g01+g11)*R1
    // v1:  g00*R0 + g01*R1
    // v2:  g10*R0 + g11*R1
    const float3 c0 = make_float3(
        -((g00+g10)*R0.x + (g01+g11)*R1.x) * wA,
        -((g00+g10)*R0.y + (g01+g11)*R1.y) * wA,
        -((g00+g10)*R0.z + (g01+g11)*R1.z) * wA);
    const float3 c1 = make_float3(
        (g00*R0.x + g01*R1.x) * wA,
        (g00*R0.y + g01*R1.y) * wA,
        (g00*R0.z + g01*R1.z) * wA);
    const float3 c2 = make_float3(
        (g10*R0.x + g11*R1.x) * wA,
        (g10*R0.y + g11*R1.y) * wA,
        (g10*R0.z + g11*R1.z) * wA);

    atomicAdd(&rhs[v0*3+0], c0.x); atomicAdd(&rhs[v0*3+1], c0.y); atomicAdd(&rhs[v0*3+2], c0.z);
    atomicAdd(&rhs[v1*3+0], c1.x); atomicAdd(&rhs[v1*3+1], c1.y); atomicAdd(&rhs[v1*3+2], c1.z);
    atomicAdd(&rhs[v2*3+0], c2.x); atomicAdd(&rhs[v2*3+1], c2.y); atomicAdd(&rhs[v2*3+2], c2.z);
}

// ============================================================================
// Bend: cotangent-weighted quadratic bending
// ============================================================================

__global__ void bend_project_kernel(
    const float* __restrict__ pos,
    const int*   __restrict__ quads,
    const float* __restrict__ weights,
    const float* __restrict__ n_rest,
    float3*      __restrict__ proj_out,
    int num_bends)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_bends) return;

    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        const int   vi = quads[e*4+i];
        const float wi = weights[e*4+i];
        accum.x += wi * pos[vi*3+0];
        accum.y += wi * pos[vi*3+1];
        accum.z += wi * pos[vi*3+2];
    }

    const float nr  = n_rest[e];
    const float len = sqrtf(accum.x*accum.x + accum.y*accum.y + accum.z*accum.z);
    float3 p;
    if (len > 1e-10f && nr > 1e-10f) {
        const float scale = nr / len;
        p = make_float3(accum.x*scale, accum.y*scale, accum.z*scale);
    } else {
        p = make_float3(0.0f, 0.0f, 0.0f);
    }

    proj_out[e] = p;
}

__global__ void accumulate_bend_rhs_kernel(
    const int*    __restrict__ quads,
    const float*  __restrict__ weights,
    const float*  __restrict__ stiffness,
    const float3* __restrict__ projections,
    float*        __restrict__ rhs,
    int num_bends, float h2)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_bends) return;

    const float k = stiffness[e] * h2;
    const float3 p = projections[e];

    for (int i = 0; i < 4; ++i) {
        const int   vi = quads[e*4+i];
        const float wi = weights[e*4+i];
        atomicAdd(&rhs[vi*3+0], k * wi * p.x);
        atomicAdd(&rhs[vi*3+1], k * wi * p.y);
        atomicAdd(&rhs[vi*3+2], k * wi * p.z);
    }
}

// ============================================================================
// Utility kernels
// ============================================================================

__global__ void clear_rhs_kernel(float* rhs, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    rhs[idx*3+0] = 0.0f; rhs[idx*3+1] = 0.0f; rhs[idx*3+2] = 0.0f;
}

__global__ void add_inertial_rhs_kernel(
    float*       __restrict__ rhs,
    const float* __restrict__ mass,
    const float* __restrict__ predict,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float m = mass[idx];
    atomicAdd(&rhs[idx*3+0], m * predict[idx*3+0]);
    atomicAdd(&rhs[idx*3+1], m * predict[idx*3+1]);
    atomicAdd(&rhs[idx*3+2], m * predict[idx*3+2]);
}

__global__ void jacobi_divide_kernel(
    float*       __restrict__ new_pos,
    const float* __restrict__ rhs,
    const float* __restrict__ jacobi_diag,
    int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float diag = jacobi_diag[idx];
    const float inv  = (diag > 1e-10f) ? (1.0f / diag) : 0.0f;
    new_pos[idx*3+0] = rhs[idx*3+0] * inv;
    new_pos[idx*3+1] = rhs[idx*3+1] * inv;
    new_pos[idx*3+2] = rhs[idx*3+2] * inv;
}

__global__ void update_velocity_kernel(
    const float* __restrict__ old_pos,
    float*       __restrict__ vel,
    const float* __restrict__ new_pos,
    int N, float dt, float damping)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    const float scale = (1.0f - damping) / dt;
    vel[idx*3+0] = (new_pos[idx*3+0] - old_pos[idx*3+0]) * scale;
    vel[idx*3+1] = (new_pos[idx*3+1] - old_pos[idx*3+1]) * scale;
    vel[idx*3+2] = (new_pos[idx*3+2] - old_pos[idx*3+2]) * scale;
}

__global__ void apply_constraints_kernel(
    float*       __restrict__ pos,
    float*       __restrict__ vel,
    const int*   __restrict__ pinned_indices,
    const float* __restrict__ target_pos,
    int num_pinned)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pinned) return;
    const int idx = pinned_indices[i];
    pos[idx*3+0] = target_pos[i*3+0];
    pos[idx*3+1] = target_pos[i*3+1];
    pos[idx*3+2] = target_pos[i*3+2];
    vel[idx*3+0] = 0.0f; vel[idx*3+1] = 0.0f; vel[idx*3+2] = 0.0f;
}

void launch_apply_constraints_kernel(
    float* d_pos, float* d_vel,
    const int* d_indices, const float* d_target,
    int num_pinned)
{
    if (num_pinned == 0) return;
    const int nb = (num_pinned + 255) / 256;
    apply_constraints_kernel<<<nb, 256>>>(d_pos, d_vel, d_indices, d_target, num_pinned);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// PDSolver implementation
// ============================================================================

PDSolver::PDSolver(const PDSolverConfig& config,
                   const ClothMesh& mesh,
                   const SimConstraints& sim_cons)
    : config_(config)
    , num_verts_(mesh.num_verts)
    , num_tris_(mesh.num_tris)
    , num_bend_cons_(sim_cons.num_bend_cons)
{
#ifndef __CUDACC__
    if (config_.use_cpu_stretch_reference) {
        cpu_stretch_ref_ = new CpuStretchReferenceSolver(mesh, sim_cons, Constraints{}, config_.dt, config_.gravity, config_.damping);
        return;
    }
#endif
    allocate_buffers(num_verts_, num_tris_, num_bend_cons_);
}

PDSolver::~PDSolver() {
#ifndef __CUDACC__
    delete cpu_stretch_ref_;
    cpu_stretch_ref_ = nullptr;
#endif
    free_buffers();
}

void PDSolver::allocate_buffers(int N, int T, int E_bend)
{
    cudaMalloc((void**)&d_predict_,  N * 3 * sizeof(float));
    cudaMalloc((void**)&d_rhs_,      N * 3 * sizeof(float));
    cudaMalloc((void**)&d_prev_pos_, N * 3 * sizeof(float));
    cudaMemset(d_prev_pos_, 0, N * 3 * sizeof(float));
    cudaMalloc((void**)&d_new_pos_,  N * 3 * sizeof(float));

    if (T > 0)
        cudaMalloc((void**)&d_tri_stretch_proj_, T * 6 * sizeof(float));
    if (E_bend > 0)
        cudaMalloc((void**)&d_bend_proj_, E_bend * sizeof(float3));
}

void PDSolver::free_buffers()
{
    auto sf = [](float*& p) { if (p) { cudaFree(p); p = nullptr; } };
    sf(d_predict_); sf(d_rhs_); sf(d_prev_pos_); sf(d_new_pos_);
    sf(d_tri_stretch_proj_); sf(d_bend_proj_);
}

void PDSolver::reset()
{
    omega_prev_ = 1.0f; omega_curr_ = 1.0f; iter_count_ = 0;
}

void PDSolver::step(ClothMesh& mesh,
                    const SimConstraints& sim_cons,
                    const Constraints& pin_cons)
{
#ifndef __CUDACC__
    if (config_.use_cpu_stretch_reference) {
        if (!cpu_stretch_ref_)
            cpu_stretch_ref_ = new CpuStretchReferenceSolver(mesh, sim_cons, pin_cons, config_.dt, config_.gravity, config_.damping);
        cpu_stretch_ref_->step(mesh, pin_cons);
        return;
    }
#endif

    const int   N   = num_verts_;
    const int   T   = num_tris_;
    const int   Eb  = num_bend_cons_;
    const float dt  = config_.dt;
    const float h2  = dt * dt;
    const int   BLK = 256;

    CUDA_CHECK(cudaMemcpy(d_prev_pos_, mesh.d_pos, N * 3 * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // Step 1: Predict
    {
        const int nb = (N + BLK - 1) / BLK;
        predict_kernel<<<nb, BLK>>>(mesh.d_pos, mesh.d_vel, d_predict_,
                                    N, dt, config_.gravity);
        CUDA_CHECK(cudaGetLastError());
    }

    // Steps 2: Local–Global Jacobi iterations
    float* d_in  = mesh.d_pos;
    float* d_out = d_new_pos_;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {

        // Local step: stretch projection
        if (T > 0 && sim_cons.d_tri_stretch_k) {
            const int nb = (T + BLK - 1) / BLK;
            tri_stretch_project_kernel<<<nb, BLK>>>(
                d_in, mesh.d_tris, mesh.d_Dm_inv, d_tri_stretch_proj_, T);
            CUDA_CHECK(cudaGetLastError());
        }

        // Local step: bend projection
        if (Eb > 0 && d_bend_proj_ && sim_cons.d_bend_quads) {
            const int nb = (Eb + BLK - 1) / BLK;
            bend_project_kernel<<<nb, BLK>>>(
                d_in,
                sim_cons.d_bend_quads,
                sim_cons.d_bend_w,
                sim_cons.d_bend_n,
                reinterpret_cast<float3*>(d_bend_proj_),
                Eb);
            CUDA_CHECK(cudaGetLastError());
        }

        // Global step: build RHS
        { const int nb = (N + BLK - 1) / BLK;
          clear_rhs_kernel<<<nb, BLK>>>(d_rhs_, N);
          CUDA_CHECK(cudaGetLastError()); }

        if (T > 0 && sim_cons.d_tri_stretch_k) {
            const int nb = (T + BLK - 1) / BLK;
            accumulate_tri_stretch_rhs_kernel<<<nb, BLK>>>(
                mesh.d_tris, mesh.d_Dm_inv, mesh.d_rest_area,
                sim_cons.d_tri_stretch_k, d_tri_stretch_proj_,
                d_rhs_, T, h2);
            CUDA_CHECK(cudaGetLastError());
        }

        if (Eb > 0 && d_bend_proj_ && sim_cons.d_bend_quads) {
            const int nb = (Eb + BLK - 1) / BLK;
            accumulate_bend_rhs_kernel<<<nb, BLK>>>(
                sim_cons.d_bend_quads,
                sim_cons.d_bend_w,
                sim_cons.d_bend_k,
                reinterpret_cast<const float3*>(d_bend_proj_),
                d_rhs_, Eb, h2);
            CUDA_CHECK(cudaGetLastError());
        }

        { const int nb = (N + BLK - 1) / BLK;
          add_inertial_rhs_kernel<<<nb, BLK>>>(d_rhs_, mesh.d_mass, d_predict_, N);
          CUDA_CHECK(cudaGetLastError()); }

        { const int nb = (N + BLK - 1) / BLK;
          jacobi_divide_kernel<<<nb, BLK>>>(d_out, d_rhs_, sim_cons.d_jacobi_diag, N);
          CUDA_CHECK(cudaGetLastError()); }

        if (pin_cons.num_pinned > 0) {
            const int nb = (pin_cons.num_pinned + BLK - 1) / BLK;
            apply_constraints_kernel<<<nb, BLK>>>(
                d_out, mesh.d_vel,
                pin_cons.d_pinned_indices, pin_cons.d_target_pos,
                pin_cons.num_pinned);
            CUDA_CHECK(cudaGetLastError());
        }

        if (config_.use_chebyshev && iter > 0)
            chebyshev_accelerate(d_in, d_out);

        std::swap(d_in, d_out);
    }

    if (d_in != mesh.d_pos)
        CUDA_CHECK(cudaMemcpy(mesh.d_pos, d_in, N * 3 * sizeof(float),
                              cudaMemcpyDeviceToDevice));

    { const int nb = (N + BLK - 1) / BLK;
      update_velocity_kernel<<<nb, BLK>>>(
          d_prev_pos_, mesh.d_vel, mesh.d_pos, N, dt, config_.damping);
      CUDA_CHECK(cudaGetLastError()); }

    if (pin_cons.num_pinned > 0) {
        const int nb = (pin_cons.num_pinned + BLK - 1) / BLK;
        apply_constraints_kernel<<<nb, BLK>>>(
            mesh.d_pos, mesh.d_vel,
            pin_cons.d_pinned_indices, pin_cons.d_target_pos,
            pin_cons.num_pinned);
        CUDA_CHECK(cudaGetLastError());
    }

    iter_count_++;
}

void PDSolver::chebyshev_accelerate(float* /*d_pos*/, float* /*d_new_pos*/)
{
    if (iter_count_ == 0) {
        omega_prev_ = 1.0f; omega_curr_ = 1.0f;
    } else if (iter_count_ == 1) {
        omega_curr_ = 2.0f / (2.0f - config_.rho * config_.rho);
    } else {
        omega_curr_ = 4.0f / (4.0f - config_.rho * config_.rho * omega_prev_);
    }
    omega_prev_ = omega_curr_;
    // TODO: apply Chebyshev blend
}
