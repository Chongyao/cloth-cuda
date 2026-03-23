#include "pd_solver.h"
#include "cloth_mesh.h"
#include "sim_constraints.h"
#include "constraints.h"
#include "utils/cuda_helper.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// ============================================================================
// CUDA Kernels
// ============================================================================

// Predict positions: y = x + h*v + h²*g  (g is acceleration, not force)
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
// Stretch: Stiefel manifold projection (nearest rotation)
// F = Ds · Dm_inv,  Ds = [x1-x0, x2-x0]  (3×2)
// R = U · V^T via analytic SVD of F^T·F (2×2 symmetric)
// Output: proj[t*6 + 0..2] = R[:,0],  proj[t*6 + 3..5] = R[:,1]
// ============================================================================

__global__ void tri_stretch_project_kernel(
    const float* __restrict__ pos,     // [N*3]
    const int*   __restrict__ tris,    // [T*3]
    const float* __restrict__ Dm_inv,  // [T*4] col-major 2×2
    float*       __restrict__ proj,    // [T*6] output rotation R
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

    // F = Ds * Dm_inv
    const float3 F0 = make_float3(ds0.x*m00 + ds1.x*m10,
                                   ds0.y*m00 + ds1.y*m10,
                                   ds0.z*m00 + ds1.z*m10);
    const float3 F1 = make_float3(ds0.x*m01 + ds1.x*m11,
                                   ds0.y*m01 + ds1.y*m11,
                                   ds0.z*m01 + ds1.z*m11);

    // A = F^T * F (2×2 symmetric)
    const float A00 = F0.x*F0.x + F0.y*F0.y + F0.z*F0.z;
    const float A01 = F0.x*F1.x + F0.y*F1.y + F0.z*F1.z;
    const float A11 = F1.x*F1.x + F1.y*F1.y + F1.z*F1.z;

    // Analytic eigendecomposition of 2×2 symmetric A
    const float tr   = (A00 + A11) * 0.5f;
    const float disc = sqrtf(fmaxf(0.0f, (A00 - A11)*(A00 - A11)*0.25f + A01*A01));
    const float lam1 = tr + disc, lam2 = tr - disc;

    float v1x, v1y;
    if (disc < 1e-10f) { v1x = 1.0f; v1y = 0.0f; }
    else {
        const float tmp = lam1 - A00;
        const float len = sqrtf(A01*A01 + tmp*tmp);
        if (len > 1e-10f) { v1x = A01/len; v1y = tmp/len; }
        else               { v1x = 1.0f;   v1y = 0.0f;   }
    }
    const float v2x = -v1y, v2y = v1x;

    const float sig1 = sqrtf(fmaxf(0.0f, lam1));
    const float sig2 = sqrtf(fmaxf(0.0f, lam2));
    const float EPS  = 1e-10f;

    float3 u1, u2;
    if (sig1 > EPS) {
        u1 = make_float3((F0.x*v1x + F1.x*v1y) / sig1,
                          (F0.y*v1x + F1.y*v1y) / sig1,
                          (F0.z*v1x + F1.z*v1y) / sig1);
    } else {
        u1 = make_float3(1.0f, 0.0f, 0.0f);
    }
    if (sig2 > EPS) {
        u2 = make_float3((F0.x*v2x + F1.x*v2y) / sig2,
                          (F0.y*v2x + F1.y*v2y) / sig2,
                          (F0.z*v2x + F1.z*v2y) / sig2);
    } else {
        const float dot = u1.x*(F0.x*v2x + F1.x*v2y)
                        + u1.y*(F0.y*v2x + F1.y*v2y)
                        + u1.z*(F0.z*v2x + F1.z*v2y);
        float3 tmp = make_float3((F0.x*v2x + F1.x*v2y) - dot*u1.x,
                                  (F0.y*v2x + F1.y*v2y) - dot*u1.y,
                                  (F0.z*v2x + F1.z*v2y) - dot*u1.z);
        const float len = sqrtf(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);
        if (len > EPS) { u2 = make_float3(tmp.x/len, tmp.y/len, tmp.z/len); }
        else {
            u2 = make_float3(-u1.y, u1.x, 0.0f);
            if (fabsf(u2.x) < EPS && fabsf(u2.y) < EPS)
                u2 = make_float3(0.0f, 1.0f, 0.0f);
        }
    }

    // R = U * V^T:  v2 = [-v1y, v1x]
    const float3 R0 = make_float3(u1.x*v1x + u2.x*(-v1y),
                                   u1.y*v1x + u2.y*(-v1y),
                                   u1.z*v1x + u2.z*(-v1y));
    const float3 R1 = make_float3(u1.x*v1y + u2.x*v1x,
                                   u1.y*v1y + u2.y*v1x,
                                   u1.z*v1y + u2.z*v1x);

    proj[t*6+0] = R0.x; proj[t*6+1] = R0.y; proj[t*6+2] = R0.z;
    proj[t*6+3] = R1.x; proj[t*6+4] = R1.y; proj[t*6+5] = R1.z;
}

// Accumulate triangle stretch RHS contributions.
// F = Ds * G, projection F → R.  RHS per vertex (derived from ∂||F-R||²/∂x_i = 0):
//   v0: -wA * ((g00+g10)*R0 + (g01+g11)*R1)
//   v1:  wA * (g00*R0 + g01*R1)
//   v2:  wA * (g10*R0 + g11*R1)
__global__ void accumulate_tri_stretch_rhs_kernel(
    const int*   __restrict__ tris,
    const float* __restrict__ Dm_inv,
    const float* __restrict__ rest_area,
    const float* __restrict__ stretch_k,
    const float* __restrict__ proj,      // [T*6]
    float*       __restrict__ rhs,
    int num_tris, float h2)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    const int v0 = tris[t*3+0], v1 = tris[t*3+1], v2 = tris[t*3+2];
    const float3 R0 = make_float3(proj[t*6+0], proj[t*6+1], proj[t*6+2]);
    const float3 R1 = make_float3(proj[t*6+3], proj[t*6+4], proj[t*6+5]);

    const float g00 = Dm_inv[t*4+0], g10 = Dm_inv[t*4+1];
    const float g01 = Dm_inv[t*4+2], g11 = Dm_inv[t*4+3];
    const float w   = stretch_k[t] * rest_area[t] * h2;

    const float3 c1 = make_float3(w*(g00*R0.x + g01*R1.x),
                                   w*(g00*R0.y + g01*R1.y),
                                   w*(g00*R0.z + g01*R1.z));
    const float3 c2 = make_float3(w*(g10*R0.x + g11*R1.x),
                                   w*(g10*R0.y + g11*R1.y),
                                   w*(g10*R0.z + g11*R1.z));
    const float3 c0 = make_float3(-(c1.x+c2.x), -(c1.y+c2.y), -(c1.z+c2.z));

    atomicAdd(&rhs[v0*3+0], c0.x); atomicAdd(&rhs[v0*3+1], c0.y); atomicAdd(&rhs[v0*3+2], c0.z);
    atomicAdd(&rhs[v1*3+0], c1.x); atomicAdd(&rhs[v1*3+1], c1.y); atomicAdd(&rhs[v1*3+2], c1.z);
    atomicAdd(&rhs[v2*3+0], c2.x); atomicAdd(&rhs[v2*3+1], c2.y); atomicAdd(&rhs[v2*3+2], c2.z);
}

// ============================================================================
// Bend: cotangent-weighted quadratic bending (Discrete Shells / DiffCloth)
//
// Local step: compute e = Σ w_i · x_i  then  p = e/|e| · n_rest
//   Output: one 3D projection vector per constraint (bend_proj[e] = p)
//
// Global step: rhs[v_i] += h² · k · w_i · p
//   Jacobi diagonal (precomputed): diag[v_i] += h² · k · w_i²
// ============================================================================

// Local step: project discrete curvature vector onto sphere of rest radius.
// Output: one float3 per constraint (the projected p vector).
__global__ void bend_project_kernel(
    const float* __restrict__ pos,      // [N*3]
    const int*   __restrict__ quads,    // [E_bend*4]  (v0,v1,v2,v3)
    const float* __restrict__ weights,  // [E_bend*4]  cotangent weights
    const float* __restrict__ n_rest,   // [E_bend]    rest curvature norms
    float3*      __restrict__ proj_out, // [E_bend]    output projection
    int num_bends)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_bends) return;

    // Compute e = Σ w_i * x_i
    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 4; ++i) {
        const int   vi = quads[e*4+i];
        const float wi = weights[e*4+i];
        accum.x += wi * pos[vi*3+0];
        accum.y += wi * pos[vi*3+1];
        accum.z += wi * pos[vi*3+2];
    }

    // Project to sphere of radius n_rest:  p = accum / |accum| * n_rest
    const float nr  = n_rest[e];
    const float len = sqrtf(accum.x*accum.x + accum.y*accum.y + accum.z*accum.z);
    float3 p;
    if (len > 1e-10f && nr > 1e-10f) {
        const float scale = nr / len;
        p = make_float3(accum.x*scale, accum.y*scale, accum.z*scale);
    } else {
        // n_rest ≈ 0 (flat cloth): drive curvature to zero → p = 0
        p = make_float3(0.0f, 0.0f, 0.0f);
    }

    proj_out[e] = p;
}

// Global step: accumulate bend RHS contributions.
// rhs[v_i] += h² · k · w_i · p
// All 4 vertices contribute (weights may be negative for wing vertices,
// which correctly pulls them in the opposite direction of p).
__global__ void accumulate_bend_rhs_kernel(
    const int*    __restrict__ quads,      // [E_bend*4]
    const float*  __restrict__ weights,    // [E_bend*4]
    const float*  __restrict__ stiffness,  // [E_bend]
    const float3* __restrict__ projections, // [E_bend]  one p per constraint
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

// ============================================================================
// External wrapper for Constraints::apply_gpu
// ============================================================================

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
    allocate_buffers(num_verts_, num_tris_, num_bend_cons_);
}

PDSolver::~PDSolver() { free_buffers(); }

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
        // One float3 projection per constraint (not per vertex as in old code)
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
    const int   N   = num_verts_;
    const int   T   = num_tris_;
    const int   Eb  = num_bend_cons_;
    const float dt  = config_.dt;
    const float h2  = dt * dt;
    const int   BLK = 256;

    // Save old position for velocity update
    CUDA_CHECK(cudaMemcpy(d_prev_pos_, mesh.d_pos, N * 3 * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // Step 1: Predict  y = x + h*v + h²*g
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

        // ---- Local step ----

        // Triangle stretch projection: F → R (nearest rotation)
        if (T > 0 && sim_cons.d_tri_stretch_k) {
            const int nb = (T + BLK - 1) / BLK;
            tri_stretch_project_kernel<<<nb, BLK>>>(
                d_in, mesh.d_tris, mesh.d_Dm_inv, d_tri_stretch_proj_, T);
            CUDA_CHECK(cudaGetLastError());
        }

        // Bend projection: e = Σ w_i x_i  →  p = e/|e| · n_rest
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

        // ---- Global step: build RHS ----

        // Clear
        { const int nb = (N + BLK - 1) / BLK;
          clear_rhs_kernel<<<nb, BLK>>>(d_rhs_, N);
          CUDA_CHECK(cudaGetLastError()); }

        // Stretch: rhs[vi] += h² · wA · (G^T R)_i
        if (T > 0 && sim_cons.d_tri_stretch_k) {
            const int nb = (T + BLK - 1) / BLK;
            accumulate_tri_stretch_rhs_kernel<<<nb, BLK>>>(
                mesh.d_tris, mesh.d_Dm_inv, mesh.d_rest_area,
                sim_cons.d_tri_stretch_k, d_tri_stretch_proj_,
                d_rhs_, T, h2);
            CUDA_CHECK(cudaGetLastError());
        }

        // Bend: rhs[vi] += h² · k · w_i · p
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

        // Inertial: rhs += M · y
        { const int nb = (N + BLK - 1) / BLK;
          add_inertial_rhs_kernel<<<nb, BLK>>>(d_rhs_, mesh.d_mass, d_predict_, N);
          CUDA_CHECK(cudaGetLastError()); }

        // Jacobi: x_new = rhs / diag
        { const int nb = (N + BLK - 1) / BLK;
          jacobi_divide_kernel<<<nb, BLK>>>(d_out, d_rhs_, sim_cons.d_jacobi_diag, N);
          CUDA_CHECK(cudaGetLastError()); }

        // Pinned constraints
        if (pin_cons.num_pinned > 0) {
            const int nb = (pin_cons.num_pinned + BLK - 1) / BLK;
            apply_constraints_kernel<<<nb, BLK>>>(
                d_out, mesh.d_vel,
                pin_cons.d_pinned_indices, pin_cons.d_target_pos,
                pin_cons.num_pinned);
            CUDA_CHECK(cudaGetLastError());
        }

        // Chebyshev acceleration
        if (config_.use_chebyshev && iter > 0)
            chebyshev_accelerate(d_in, d_out);

        std::swap(d_in, d_out);
    }

    // Ensure final result is in mesh.d_pos
    if (d_in != mesh.d_pos)
        CUDA_CHECK(cudaMemcpy(mesh.d_pos, d_in, N * 3 * sizeof(float),
                              cudaMemcpyDeviceToDevice));

    // Step 3: Velocity update  v = (x_new - x_old) / dt
    { const int nb = (N + BLK - 1) / BLK;
      update_velocity_kernel<<<nb, BLK>>>(
          d_prev_pos_, mesh.d_vel, mesh.d_pos, N, dt, config_.damping);
      CUDA_CHECK(cudaGetLastError()); }

    // Re-enforce pinned constraints on velocity
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
    // Update omega sequence (Chebyshev semi-iterative method)
    if (iter_count_ == 0) {
        omega_prev_ = 1.0f; omega_curr_ = 1.0f;
    } else if (iter_count_ == 1) {
        omega_curr_ = 2.0f / (2.0f - config_.rho * config_.rho);
    } else {
        omega_curr_ = 4.0f / (4.0f - config_.rho * config_.rho * omega_prev_);
    }
    omega_prev_ = omega_curr_;
    // TODO Phase 4: apply blend  x = omega*(x_jacobi - x_prev) + x_prev
    //   Requires d_cheby_prev_ [N*3] buffer + blend kernel.
}
