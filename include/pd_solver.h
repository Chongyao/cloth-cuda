#pragma once

// PD Solver configuration
struct PDSolverConfig {
    int max_iterations = 50;          // Max local-global iterations per frame
    float tolerance = 1e-4f;          // Convergence tolerance
    bool use_chebyshev = true;        // Enable Chebyshev acceleration
    float rho = 0.9f;                 // Spectral radius estimate for Chebyshev
    float gravity = -9.8f;            // Gravity (m/s^2)
    float dt = 0.01f;                 // Time step
    float stretch_stiffness = 1.0f;   // Stretch constraint weight
    float bend_stiffness = 0.0f;      // Bend constraint weight (Phase 4)
    float damping = 0.0f;             // Velocity damping (0 = no damping, 0.99 = heavy)
};

// Forward declarations - definitions in mesh.h and constraints.h
struct ClothMesh;
struct Constraints;

// GPU-based Projective Dynamics Solver
class PDSolver {
public:
    PDSolver(const PDSolverConfig& config, ClothMesh& mesh);
    ~PDSolver();

    // Execute one simulation step
    void step(ClothMesh& mesh, const Constraints& cons);

    // Reset Chebyshev state (call when constraints change)
    void reset();

private:
    PDSolverConfig config_;
    int num_verts_;
    int num_stretch_cons_;
    int num_bend_cons_;

    // Temporary GPU buffers
    float* d_predict_ = nullptr;      // Predicted positions (inertial term)
    float* d_rhs_ = nullptr;          // RHS for global step
    float* d_prev_pos_ = nullptr;     // Old position saved at step start (for velocity update)
    float* d_new_pos_ = nullptr;      // Ping-pong buffer for Jacobi iterations
    float* d_stretch_proj_ = nullptr; // Stretch constraint projections
    float* d_bend_proj_ = nullptr;    // Bend constraint projections (Phase 4)

    // Chebyshev state
    float omega_prev_ = 1.0f;
    float omega_curr_ = 1.0f;
    int iter_count_ = 0;

    // CUDA kernels (wrappers)
    void predict_positions(float* d_pos, float* d_vel);
    void local_step_stretch(const float* d_pos);
    void global_step_jacobi(float* d_pos, float* d_new_pos);
    void update_velocity(float* d_pos, float* d_vel, const float* d_new_pos);
    void chebyshev_accelerate(float* d_pos, float* d_new_pos);

    void allocate_buffers(int N, int E_stretch, int E_bend);
    void free_buffers();
};

// External kernel wrappers for Constraints::apply_gpu
void launch_apply_constraints_kernel(
    float* d_pos, float* d_vel,
    const int* d_indices, const float* d_target,
    int num_pinned);
