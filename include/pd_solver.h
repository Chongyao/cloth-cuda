#pragma once

// PD Solver configuration
struct PDSolverConfig {
    int   max_iterations     = 50;    // local-global iterations per frame
    float tolerance          = 1e-4f; // convergence tolerance (unused: Jacobi runs fixed iters)
    bool  use_chebyshev      = true;  // enable Chebyshev acceleration
    float rho                = 0.9f;  // spectral radius estimate for Chebyshev
    float gravity            = -9.8f; // gravitational acceleration (m/s²)
    float dt                 = 0.01f; // time step
    float stretch_stiffness  = 1.0f;  // triangle stretch constraint weight
    float bend_stiffness     = 0.0f;  // bend constraint weight
    float damping            = 0.0f;  // velocity damping (0 = none, 1 = full)
};

// Forward declarations
struct ClothMesh;
struct SimConstraints;
struct Constraints;

// GPU-based Projective Dynamics solver.
//
// Usage:
//   PDSolver solver(config, mesh, sim_cons);   // allocates temp GPU buffers
//   // per frame:
//   solver.step(mesh, sim_cons, pin_cons);
class PDSolver {
public:
    PDSolver(const PDSolverConfig& config,
             const ClothMesh& mesh,
             const SimConstraints& sim_cons);
    ~PDSolver();

    // Execute one simulation step (predict → local → global → velocity update).
    void step(ClothMesh& mesh,
              const SimConstraints& sim_cons,
              const Constraints& pin_cons);

    // Reset Chebyshev state (call when constraints change at runtime).
    void reset();

private:
    PDSolverConfig config_;
    int num_verts_;
    int num_tris_;
    int num_bend_cons_;

    // ---- Per-step temporary GPU buffers ----
    float* d_predict_         = nullptr;  // [N*3] inertial prediction y
    float* d_rhs_             = nullptr;  // [N*3] global step RHS accumulator
    float* d_prev_pos_        = nullptr;  // [N*3] position at frame start (for v update)
    float* d_new_pos_         = nullptr;  // [N*3] Jacobi ping-pong output
    float* d_tri_stretch_proj_ = nullptr; // [T*6] Stiefel projection R per triangle
    float* d_bend_proj_       = nullptr;  // [E_bend*4*3] bend projections

    // ---- Chebyshev state ----
    float omega_prev_  = 1.0f;
    float omega_curr_  = 1.0f;
    int   iter_count_  = 0;

    void allocate_buffers(int N, int T, int E_bend);
    void free_buffers();
    void chebyshev_accelerate(float* d_pos, float* d_new_pos);
};

// Kernel wrapper used by Constraints::apply_gpu (implemented in pd_solver.cu)
void launch_apply_constraints_kernel(
    float* d_pos, float* d_vel,
    const int* d_indices, const float* d_target,
    int num_pinned);
