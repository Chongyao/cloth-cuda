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

// Bend constraint projection: rotate the two triangle "wings" around the shared
// edge so the dihedral angle matches rest_angle.
// Each bend constraint stores quad (v0,v1,v2,v3): v0-v1 = shared edge,
// v2 = opposite vertex in tri A, v3 = opposite vertex in tri B.
__global__ void bend_project_kernel(
    const float* __restrict__ pos,
    const int*   __restrict__ quads,        // [E_bend*4]
    const float* __restrict__ rest_angles,  // [E_bend]
    float3*      __restrict__ projections,  // [E_bend*4] output
    int num_bends)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_bends) return;

    int v0 = quads[e*4+0], v1 = quads[e*4+1];
    int v2 = quads[e*4+2], v3 = quads[e*4+3];

    float3 p0 = make_float3(pos[v0*3], pos[v0*3+1], pos[v0*3+2]);
    float3 p1 = make_float3(pos[v1*3], pos[v1*3+1], pos[v1*3+2]);
    float3 p2 = make_float3(pos[v2*3], pos[v2*3+1], pos[v2*3+2]);
    float3 p3 = make_float3(pos[v3*3], pos[v3*3+1], pos[v3*3+2]);

    // Default: no change
    projections[e*4+0] = p0;
    projections[e*4+1] = p1;
    projections[e*4+2] = p2;
    projections[e*4+3] = p3;

    // Shared edge axis
    float3 edge = {p1.x-p0.x, p1.y-p0.y, p1.z-p0.z};
    float edge_len = sqrtf(edge.x*edge.x + edge.y*edge.y + edge.z*edge.z);
    if (edge_len < 1e-10f) return;
    float3 ax = {edge.x/edge_len, edge.y/edge_len, edge.z/edge_len};

    // Perpendicular components of v2 and v3 relative to the edge line
    auto perp_from_edge = [&](float3 p) -> float3 {
        float t = (p.x-p0.x)*ax.x + (p.y-p0.y)*ax.y + (p.z-p0.z)*ax.z;
        float3 foot = {p0.x+t*ax.x, p0.y+t*ax.y, p0.z+t*ax.z};
        return {p.x-foot.x, p.y-foot.y, p.z-foot.z};
    };

    float3 r2 = perp_from_edge(p2);
    float3 r3 = perp_from_edge(p3);
    float r2_len = sqrtf(r2.x*r2.x + r2.y*r2.y + r2.z*r2.z);
    float r3_len = sqrtf(r3.x*r3.x + r3.y*r3.y + r3.z*r3.z);
    if (r2_len < 1e-10f || r3_len < 1e-10f) return;

    float3 r2h = {r2.x/r2_len, r2.y/r2_len, r2.z/r2_len};
    float3 r3h = {r3.x/r3_len, r3.y/r3_len, r3.z/r3_len};

    // Current dihedral angle (signed, around ax)
    float cos_t = fmaxf(-1.0f, fminf(1.0f,
        r2h.x*r3h.x + r2h.y*r3h.y + r2h.z*r3h.z));
    float3 cr = {r2h.y*r3h.z - r2h.z*r3h.y,
                 r2h.z*r3h.x - r2h.x*r3h.z,
                 r2h.x*r3h.y - r2h.y*r3h.x};
    float sin_t = cr.x*ax.x + cr.y*ax.y + cr.z*ax.z;
    float theta = atan2f(sin_t, cos_t);

    // Wrap angle difference to (-π, π] to avoid discontinuity at ±π
    float diff = theta - rest_angles[e];
    if (diff >  3.14159265f) diff -= 6.28318530f;
    if (diff < -3.14159265f) diff += 6.28318530f;

    // Cap per-step correction: Jacobi-PD diverges for large nonlinear rotations.
    // Bending stiffness should be << stretch stiffness for Jacobi convergence.
    const float MAX_HALF_DELTA = 0.17453f;  // 10° per iteration
    float half_delta = fmaxf(-MAX_HALF_DELTA, fminf(MAX_HALF_DELTA, diff * 0.5f));

    // Rodrigues rotation around ax (r is perpendicular to ax, so ax·r = 0)
    auto rotate_perp = [&](float3 r, float angle) -> float3 {
        float ca = cosf(angle), sa = sinf(angle);
        float3 cr2 = {ax.y*r.z - ax.z*r.y, ax.z*r.x - ax.x*r.z, ax.x*r.y - ax.y*r.x};
        return {ca*r.x + sa*cr2.x, ca*r.y + sa*cr2.y, ca*r.z + sa*cr2.z};
    };

    // r2 rotates by -half_delta, r3 by +half_delta (symmetric correction)
    float3 r2_new = rotate_perp(r2, -half_delta);
    float3 r3_new = rotate_perp(r3,  half_delta);

    // Recompute foot positions
    auto foot = [&](float3 p) -> float3 {
        float t = (p.x-p0.x)*ax.x + (p.y-p0.y)*ax.y + (p.z-p0.z)*ax.z;
        return {p0.x+t*ax.x, p0.y+t*ax.y, p0.z+t*ax.z};
    };
    float3 f2 = foot(p2), f3 = foot(p3);

    // Edge vertices (v0, v1) project to their current positions
    projections[e*4+2] = {f2.x+r2_new.x, f2.y+r2_new.y, f2.z+r2_new.z};
    projections[e*4+3] = {f3.x+r3_new.x, f3.y+r3_new.y, f3.z+r3_new.z};
}

// Accumulate bend constraint RHS contributions — only v2/v3 (wing vertices).
// v0/v1 (shared edge) project to their current positions, so including them
// would add a spurious "pull toward current pos" term and bloat the diagonal.
__global__ void accumulate_bend_rhs_kernel(
    const int*   __restrict__ quads,
    const float3* __restrict__ projections,
    const float* __restrict__ stiffness,
    float*       __restrict__ rhs,
    int num_bends,
    float h2)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_bends) return;

    int v2 = quads[e*4+2], v3 = quads[e*4+3];
    float w = stiffness[e] * h2;

    float3 p2 = projections[e*4+2], p3 = projections[e*4+3];

    atomicAdd(&rhs[v2*3+0], w*p2.x); atomicAdd(&rhs[v2*3+1], w*p2.y); atomicAdd(&rhs[v2*3+2], w*p2.z);
    atomicAdd(&rhs[v3*3+0], w*p3.x); atomicAdd(&rhs[v3*3+1], w*p3.y); atomicAdd(&rhs[v3*3+2], w*p3.z);
}

// ============================================================================
// Triangle-based stretch constraints (Stiefel projection)
// ============================================================================

// Triangle stretch projection: project deformation gradient F to nearest rotation R
// F = Ds * Dm_inv where Ds = [x0-x2, x1-x2]
// R = U * V^T via SVD of F
__global__ void tri_stretch_project_kernel(
    const float* __restrict__ pos,        // [N*3]
    const int*   __restrict__ tris,       // [T*3]
    const float* __restrict__ Dm_inv,     // [T*4] col-major 2x2
    float*       __restrict__ proj,       // [T*6] output: R = [col0(3), col1(3)]
    int num_tris)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    // Read triangle vertices
    int v0 = tris[t*3+0];
    int v1 = tris[t*3+1];
    int v2 = tris[t*3+2];

    // Load positions
    float3 p0 = make_float3(pos[v0*3+0], pos[v0*3+1], pos[v0*3+2]);
    float3 p1 = make_float3(pos[v1*3+0], pos[v1*3+1], pos[v1*3+2]);
    float3 p2 = make_float3(pos[v2*3+0], pos[v2*3+1], pos[v2*3+2]);

    // Ds = [x1-x0, x2-x0] (two columns) - must match Dm = [X1-X0, X2-X0]
    float3 ds0 = make_float3(p1.x-p0.x, p1.y-p0.y, p1.z-p0.z);
    float3 ds1 = make_float3(p2.x-p0.x, p2.y-p0.y, p2.z-p0.z);

    // Load Dm_inv (col-major): [m00, m10, m01, m11]
    float m00 = Dm_inv[t*4+0];
    float m10 = Dm_inv[t*4+1];
    float m01 = Dm_inv[t*4+2];
    float m11 = Dm_inv[t*4+3];

    // F = Ds * Dm_inv = [ds0*m00 + ds1*m10, ds0*m01 + ds1*m11]
    float3 F0 = make_float3(
        ds0.x*m00 + ds1.x*m10,
        ds0.y*m00 + ds1.y*m10,
        ds0.z*m00 + ds1.z*m10);
    float3 F1 = make_float3(
        ds0.x*m01 + ds1.x*m11,
        ds0.y*m01 + ds1.y*m11,
        ds0.z*m01 + ds1.z*m11);

    // SVD of F (3x2): via A = F^T * F (2x2 symmetric)
    // A = [F0·F0, F0·F1; F0·F1, F1·F1]
    float A00 = F0.x*F0.x + F0.y*F0.y + F0.z*F0.z;
    float A01 = F0.x*F1.x + F0.y*F1.y + F0.z*F1.z;
    float A11 = F1.x*F1.x + F1.y*F1.y + F1.z*F1.z;

    // Analytic eigendecomposition of 2x2 symmetric A
    float tr = (A00 + A11) * 0.5f;
    float disc = sqrtf(fmaxf(0.0f, (A00 - A11) * (A00 - A11) * 0.25f + A01 * A01));
    float lambda1 = tr + disc;
    float lambda2 = tr - disc;

    // Eigenvectors of A (V columns)
    float v1x, v1y, v2x, v2y;
    if (disc < 1e-10f) {
        // Degenerate: A is diagonal/isotropic
        v1x = 1.0f; v1y = 0.0f;
        v2x = 0.0f; v2y = 1.0f;
    } else {
        // v1 = normalize([A01, lambda1 - A00])
        float tmp = lambda1 - A00;
        float len1 = sqrtf(A01*A01 + tmp*tmp);
        if (len1 > 1e-10f) {
            v1x = A01 / len1;
            v1y = tmp / len1;
        } else {
            v1x = 1.0f; v1y = 0.0f;
        }
        // v2 is perpendicular
        v2x = -v1y;
        v2y = v1x;
    }

    // Singular values
    float sigma1 = sqrtf(fmaxf(0.0f, lambda1));
    float sigma2 = sqrtf(fmaxf(0.0f, lambda2));

    // U columns: ui = (F * vi) / sigma_i
    float3 u1, u2;
    const float EPS = 1e-10f;

    if (sigma1 > EPS) {
        u1 = make_float3(
            (F0.x*v1x + F1.x*v1y) / sigma1,
            (F0.y*v1x + F1.y*v1y) / sigma1,
            (F0.z*v1x + F1.z*v1y) / sigma1);
    } else {
        u1 = make_float3(1.0f, 0.0f, 0.0f);
    }

    if (sigma2 > EPS) {
        u2 = make_float3(
            (F0.x*v2x + F1.x*v2y) / sigma2,
            (F0.y*v2x + F1.y*v2y) / sigma2,
            (F0.z*v2x + F1.z*v2y) / sigma2);
    } else {
        // Gram-Schmidt orthogonalize against u1
        float dot = u1.x * (F0.x*v2x + F1.x*v2y)
                  + u1.y * (F0.y*v2x + F1.y*v2y)
                  + u1.z * (F0.z*v2x + F1.z*v2y);
        float3 tmp = make_float3(
            (F0.x*v2x + F1.x*v2y) - dot * u1.x,
            (F0.y*v2x + F1.y*v2y) - dot * u1.y,
            (F0.z*v2x + F1.z*v2y) - dot * u1.z);
        float len = sqrtf(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);
        if (len > EPS) {
            u2 = make_float3(tmp.x/len, tmp.y/len, tmp.z/len);
        } else {
            u2 = make_float3(-u1.y, u1.x, 0.0f);  // Perpendicular fallback
            if (fabsf(u2.x) < EPS && fabsf(u2.y) < EPS) {
                u2 = make_float3(0.0f, 1.0f, 0.0f);
            }
        }
    }

    // R = U * V^T = [u1, u2] * [v1, v2]^T = u1*v1^T + u2*v2^T
    // V^T rows are v1^T and v2^T, so:
    //   R[:,0] = u1*v1x + u2*v2x
    //   R[:,1] = u1*v1y + u2*v2y
    // But v2 is perpendicular to v1: v2 = [-v1y, v1x]
    // So: v2x = -v1y, v2y = v1x
    float3 R0 = make_float3(
        u1.x*v1x + u2.x*(-v1y),
        u1.y*v1x + u2.y*(-v1y),
        u1.z*v1x + u2.z*(-v1y));
    float3 R1 = make_float3(
        u1.x*v1y + u2.x*v1x,
        u1.y*v1y + u2.y*v1x,
        u1.z*v1y + u2.z*v1x);

    // Write output: proj[t*6 + 0..2] = R0, proj[t*6 + 3..5] = R1
    proj[t*6+0] = R0.x; proj[t*6+1] = R0.y; proj[t*6+2] = R0.z;
    proj[t*6+3] = R1.x; proj[t*6+4] = R1.y; proj[t*6+5] = R1.z;
}

// Accumulate triangle stretch contributions to RHS
// With Ds = [x1-x0, x2-x0] and F = Ds * Dm_inv, we want F = R
// RHS contributions (derived from gradient of ||F - R||^2):
//   v0: -w*A * ((m00+m10)*R0 + (m01+m11)*R1)  [negative sum]
//   v1:  w*A * (m00*R0 + m01*R1)
//   v2:  w*A * (m10*R0 + m11*R1)
__global__ void accumulate_tri_stretch_rhs_kernel(
    const int*   __restrict__ tris,       // [T*3]
    const float* __restrict__ Dm_inv,     // [T*4]
    const float* __restrict__ rest_area,  // [T]
    const float* __restrict__ stretch_k,  // [T]
    const float* __restrict__ proj,       // [T*6] = [R0(3), R1(3)]
    float*       __restrict__ rhs,        // [N*3]
    int num_tris,
    float h2)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tris) return;

    // Vertices
    int v0 = tris[t*3+0];
    int v1 = tris[t*3+1];
    int v2 = tris[t*3+2];

    // Load R columns
    float3 R0 = make_float3(proj[t*6+0], proj[t*6+1], proj[t*6+2]);
    float3 R1 = make_float3(proj[t*6+3], proj[t*6+4], proj[t*6+5]);

    // Dm_inv (col-major): G = [g00 g01; g10 g11] where g00=m00, g10=m10, g01=m01, g11=m11
    float g00 = Dm_inv[t*4+0];
    float g10 = Dm_inv[t*4+1];
    float g01 = Dm_inv[t*4+2];
    float g11 = Dm_inv[t*4+3];

    // Weight: wA = stiffness * area
    float wA = stretch_k[t] * rest_area[t];
    float w = wA * h2;

    // v1 contribution: w * (g00*R0 + g01*R1)
    float3 c1 = make_float3(
        w * (g00*R0.x + g01*R1.x),
        w * (g00*R0.y + g01*R1.y),
        w * (g00*R0.z + g01*R1.z));

    // v2 contribution: w * (g10*R0 + g11*R1)
    float3 c2 = make_float3(
        w * (g10*R0.x + g11*R1.x),
        w * (g10*R0.y + g11*R1.y),
        w * (g10*R0.z + g11*R1.z));

    // v0 contribution: -(c1 + c2) = -w * ((g00+g10)*R0 + (g01+g11)*R1)
    float3 c0 = make_float3(-(c1.x + c2.x), -(c1.y + c2.y), -(c1.z + c2.z));

    atomicAdd(&rhs[v0*3+0], c0.x); atomicAdd(&rhs[v0*3+1], c0.y); atomicAdd(&rhs[v0*3+2], c0.z);
    atomicAdd(&rhs[v1*3+0], c1.x); atomicAdd(&rhs[v1*3+1], c1.y); atomicAdd(&rhs[v1*3+2], c1.z);
    atomicAdd(&rhs[v2*3+0], c2.x); atomicAdd(&rhs[v2*3+1], c2.y); atomicAdd(&rhs[v2*3+2], c2.z);
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
    , num_tris_(mesh.num_tris)
    , num_stretch_cons_(mesh.num_stretch_cons)  // Deprecated
    , num_bend_cons_(mesh.num_bend_cons)
{
    allocate_buffers(num_verts_, num_tris_, num_bend_cons_);

    // Check if triangle-based stretch is available, warn if neither stretch type is available
    if (!mesh.d_tri_stretch_k && !mesh.d_stretch_edges) {
        fprintf(stderr, "PDSolver WARNING: No stretch constraints (triangle or edge) available\n");
        fprintf(stderr, "  Call mesh.build_tri_stretch() and mesh.upload_to_gpu() first\n");
    }
}

PDSolver::~PDSolver()
{
    free_buffers();
}

void PDSolver::allocate_buffers(int N, int T, int E_bend)
{
    cudaMalloc((void**)&d_predict_, N * 3 * sizeof(float));
    cudaMalloc((void**)&d_rhs_, N * 3 * sizeof(float));
    cudaMalloc((void**)&d_prev_pos_, N * 3 * sizeof(float));
    cudaMemset(d_prev_pos_, 0, N * 3 * sizeof(float));
    cudaMalloc((void**)&d_new_pos_, N * 3 * sizeof(float));

    if (T > 0) {
        // Each triangle stores a 3x2 projection matrix R (6 floats)
        cudaMalloc((void**)&d_tri_stretch_proj_, T * 6 * sizeof(float));
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
    if (d_tri_stretch_proj_) { cudaFree(d_tri_stretch_proj_); d_tri_stretch_proj_ = nullptr; }
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

    const int T = num_tris_;
    const int E_bend = num_bend_cons_;

    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        // --- Local Step ---
        // Triangle-based stretch (Stiefel projection)
        if (T > 0 && mesh.d_tri_stretch_k) {
            int num_blocks = (T + block_size - 1) / block_size;
            tri_stretch_project_kernel<<<num_blocks, block_size>>>(
                d_pos_in,
                mesh.d_tris,
                mesh.d_Dm_inv,
                d_tri_stretch_proj_,
                T);
            CUDA_CHECK(cudaGetLastError());
        }

        if (E_bend > 0 && d_bend_proj_ && mesh.d_bend_quads) {
            int num_blocks = (E_bend + block_size - 1) / block_size;
            bend_project_kernel<<<num_blocks, block_size>>>(
                d_pos_in,
                mesh.d_bend_quads,
                mesh.d_bend_rest,
                reinterpret_cast<float3*>(d_bend_proj_),
                E_bend);
            CUDA_CHECK(cudaGetLastError());
        }

        // --- Global Step: RHS = M*y + h²*(stretch + bend contributions) ---
        {
            int num_blocks = (N + block_size - 1) / block_size;
            clear_rhs_kernel<<<num_blocks, block_size>>>(d_rhs_, N);
            CUDA_CHECK(cudaGetLastError());
        }

        // Accumulate triangle stretch contributions
        if (T > 0 && mesh.d_tri_stretch_k) {
            int num_blocks = (T + block_size - 1) / block_size;
            accumulate_tri_stretch_rhs_kernel<<<num_blocks, block_size>>>(
                mesh.d_tris,
                mesh.d_Dm_inv,
                mesh.d_rest_area,
                mesh.d_tri_stretch_k,
                d_tri_stretch_proj_,
                d_rhs_, T, h2);
            CUDA_CHECK(cudaGetLastError());
        }

        if (E_bend > 0 && d_bend_proj_ && mesh.d_bend_quads) {
            int num_blocks = (E_bend + block_size - 1) / block_size;
            accumulate_bend_rhs_kernel<<<num_blocks, block_size>>>(
                mesh.d_bend_quads,
                reinterpret_cast<const float3*>(d_bend_proj_),
                mesh.d_bend_k,
                d_rhs_, E_bend, h2);
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
            N, dt, config_.damping);
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
