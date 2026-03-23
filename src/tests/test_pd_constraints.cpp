#include "cloth_mesh.h"
#include "mesh_topology.h"
#include "sim_constraints.h"
#include "constraints.h"
#include "mesh_generator.h"
#include "pd_solver.h"
#include "tests/test_framework.h"

#include <cmath>
#include <cstdio>
#include <vector>

#ifdef CUDA_MS_HAVE_CUDA
#include <cuda_runtime.h>
#endif

// ---- GPU readback helpers ----

static float max_displacement(const ClothMesh& mesh, const std::vector<float>& rest_flat)
{
    std::vector<float> h_pos(mesh.num_verts * 3);
#ifdef CUDA_MS_HAVE_CUDA
    cudaMemcpy(h_pos.data(), mesh.d_pos,
               mesh.num_verts * 3 * sizeof(float), cudaMemcpyDeviceToHost);
#else
    for (int i = 0; i < mesh.num_verts; ++i) {
        h_pos[i*3+0] = mesh.rest_pos[i](0);
        h_pos[i*3+1] = mesh.rest_pos[i](1);
        h_pos[i*3+2] = mesh.rest_pos[i](2);
    }
#endif
    float max_d = 0.0f;
    for (int i = 0; i < mesh.num_verts; ++i) {
        float dx = h_pos[i*3+0] - rest_flat[i*3+0];
        float dy = h_pos[i*3+1] - rest_flat[i*3+1];
        float dz = h_pos[i*3+2] - rest_flat[i*3+2];
        max_d = std::max(max_d, std::sqrt(dx*dx + dy*dy + dz*dz));
    }
    return max_d;
}

static float max_velocity(const ClothMesh& mesh)
{
    std::vector<float> h_vel(mesh.num_verts * 3, 0.0f);
#ifdef CUDA_MS_HAVE_CUDA
    cudaMemcpy(h_vel.data(), mesh.d_vel,
               mesh.num_verts * 3 * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    float max_v = 0.0f;
    for (int i = 0; i < mesh.num_verts; ++i) {
        float vx = h_vel[i*3+0], vy = h_vel[i*3+1], vz = h_vel[i*3+2];
        max_v = std::max(max_v, std::sqrt(vx*vx + vy*vy + vz*vz));
    }
    return max_v;
}

static float kinetic_energy(const ClothMesh& mesh)
{
    std::vector<float> h_vel(mesh.num_verts * 3, 0.0f);
    std::vector<float> h_mass(mesh.num_verts, 0.0f);
#ifdef CUDA_MS_HAVE_CUDA
    cudaMemcpy(h_vel.data(),  mesh.d_vel,  mesh.num_verts * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mass.data(), mesh.d_mass, mesh.num_verts     * sizeof(float), cudaMemcpyDeviceToHost);
#else
    for (int i = 0; i < mesh.num_verts; ++i) h_mass[i] = mesh.mass[i];
#endif
    float ke = 0.0f;
    for (int i = 0; i < mesh.num_verts; ++i) {
        float vx = h_vel[i*3+0], vy = h_vel[i*3+1], vz = h_vel[i*3+2];
        ke += 0.5f * h_mass[i] * (vx*vx + vy*vy + vz*vz);
    }
    return ke;
}

static void set_positions(ClothMesh& mesh, const std::vector<float>& pos)
{
#ifdef CUDA_MS_HAVE_CUDA
    cudaMemcpy(mesh.d_pos, pos.data(),
               mesh.num_verts * 3 * sizeof(float), cudaMemcpyHostToDevice);
#else
    for (int i = 0; i < mesh.num_verts; ++i) {
        mesh.rest_pos[i](0) = pos[i*3+0];
        mesh.rest_pos[i](1) = pos[i*3+1];
        mesh.rest_pos[i](2) = pos[i*3+2];
    }
#endif
}

static void zero_velocities(ClothMesh& mesh)
{
#ifdef CUDA_MS_HAVE_CUDA
    cudaMemset(mesh.d_vel, 0, mesh.num_verts * 3 * sizeof(float));
#endif
}

// Flatten CPU rest positions into a float array
static std::vector<float> rest_flat(const ClothMesh& mesh)
{
    std::vector<float> v(mesh.num_verts * 3);
    for (int i = 0; i < mesh.num_verts; ++i) {
        v[i*3+0] = mesh.rest_pos[i](0);
        v[i*3+1] = mesh.rest_pos[i](1);
        v[i*3+2] = mesh.rest_pos[i](2);
    }
    return v;
}

// ---- Tests ----

int main()
{
    printf("=== PD Constraint Tests ===\n\n");

    // Test 1: Zero gravity + rest shape → should remain static
    SECTION("Test 1: Zero gravity + rest shape = static");
    {
        ClothMesh mesh;
        generate_square_cloth(10, 10, 0.1f, 3, mesh);
        mesh.precompute_rest_state(0.1f);

        MeshTopology topo = MeshTopology::build(mesh);
        SimConstraints sim_cons;
        sim_cons.build_stretch(mesh, 100.0f);
        sim_cons.build_bend(mesh, topo, 1.0f);
        mesh.upload_to_gpu();
        sim_cons.upload_to_gpu();
        sim_cons.precompute_jacobi_diag(mesh, 0.01f);

        Constraints pin_cons;
        pin_cons.pin_corners(mesh, 10);
        pin_cons.upload_to_gpu();

        PDSolverConfig cfg;
        cfg.dt = 0.01f; cfg.gravity = 0.0f; cfg.max_iterations = 100;
        PDSolver solver(cfg, mesh, sim_cons);

        auto rp = rest_flat(mesh);
        set_positions(mesh, rp);
        zero_velocities(mesh);

        for (int s = 0; s < 100; ++s)
            solver.step(mesh, sim_cons, pin_cons);

        float max_v = max_velocity(mesh);
        float ke    = kinetic_energy(mesh);
        printf("  Max velocity: %e, KE: %e\n", max_v, ke);
        CHECK(max_v < 1e-2f);
        CHECK(ke    < 1e-4f);
    }

    // Test 2: High stretch stiffness + cylindrical bend → stays nearly isometric
    SECTION("Test 2: Isometric bending, no bend constraints = slow drift");
    {
        ClothMesh mesh;
        generate_square_cloth(10, 10, 0.1f, 0, mesh);
        mesh.precompute_rest_state(0.1f);

        SimConstraints sim_cons;
        sim_cons.build_stretch(mesh, 1000.0f);
        mesh.upload_to_gpu();
        sim_cons.upload_to_gpu();
        sim_cons.precompute_jacobi_diag(mesh, 0.01f);

        Constraints pin_cons;  // no pins
        pin_cons.upload_to_gpu();

        PDSolverConfig cfg;
        cfg.dt = 0.01f; cfg.gravity = 0.0f; cfg.max_iterations = 50;
        PDSolver solver(cfg, mesh, sim_cons);

        std::vector<float> init_pos(mesh.num_verts * 3);
        const float R = 1.0f;
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 10; ++j) {
                int idx = i*10+j;
                float th = (i-4.5f)*0.1f/R;
                init_pos[idx*3+0] = mesh.rest_pos[idx](0);
                init_pos[idx*3+1] = R*(1.0f-std::cos(th));
                init_pos[idx*3+2] = R*std::sin(th);
            }
        set_positions(mesh, init_pos);
        zero_velocities(mesh);

        for (int s = 0; s < 50; ++s)
            solver.step(mesh, sim_cons, pin_cons);

        float max_v = max_velocity(mesh);
        printf("  Max velocity: %e\n", max_v);
        CHECK(max_v < 5e-2f);
    }

    // Test 3: No stretch constraints, bend only + uniform scale → stays static
    SECTION("Test 3: No stretch, uniform scale = static");
    {
        ClothMesh mesh;
        generate_square_cloth(10, 10, 0.1f, 0, mesh);
        mesh.precompute_rest_state(0.1f);

        MeshTopology topo = MeshTopology::build(mesh);
        SimConstraints sim_cons;
        sim_cons.build_bend(mesh, topo, 1.0f);
        mesh.upload_to_gpu();
        sim_cons.upload_to_gpu();
        sim_cons.precompute_jacobi_diag(mesh, 0.01f);

        Constraints pin_cons;
        pin_cons.upload_to_gpu();

        PDSolverConfig cfg;
        cfg.dt = 0.01f; cfg.gravity = 0.0f; cfg.max_iterations = 50;
        PDSolver solver(cfg, mesh, sim_cons);

        std::vector<float> init_pos(mesh.num_verts * 3);
        for (int i = 0; i < mesh.num_verts; ++i) {
            init_pos[i*3+0] = mesh.rest_pos[i](0)*1.5f;
            init_pos[i*3+1] = mesh.rest_pos[i](1)*1.5f;
            init_pos[i*3+2] = mesh.rest_pos[i](2)*1.5f;
        }
        set_positions(mesh, init_pos);
        zero_velocities(mesh);

        for (int s = 0; s < 50; ++s)
            solver.step(mesh, sim_cons, pin_cons);

        float max_v = max_velocity(mesh);
        printf("  Max velocity: %e\n", max_v);
        CHECK(max_v < 1e-2f);
    }

    // Test 4: Stretched initial state → energy stays bounded
    SECTION("Test 4: Stretched initial + stretch constraints → bounded energy");
    {
        ClothMesh mesh;
        generate_square_cloth(10, 10, 0.1f, 0, mesh);
        mesh.precompute_rest_state(0.1f);

        SimConstraints sim_cons;
        sim_cons.build_stretch(mesh, 100.0f);
        mesh.upload_to_gpu();
        sim_cons.upload_to_gpu();
        sim_cons.precompute_jacobi_diag(mesh, 0.005f);

        Constraints pin_cons;
        pin_cons.pin_corners(mesh, 10);
        pin_cons.upload_to_gpu();

        PDSolverConfig cfg;
        cfg.dt = 0.005f; cfg.gravity = 0.0f; cfg.max_iterations = 200;
        PDSolver solver(cfg, mesh, sim_cons);

        auto rp = rest_flat(mesh);
        std::vector<float> init_pos(mesh.num_verts * 3);
        for (int i = 0; i < mesh.num_verts; ++i) {
            init_pos[i*3+0] = mesh.rest_pos[i](0)*1.5f;
            init_pos[i*3+1] = mesh.rest_pos[i](1)*1.5f;
            init_pos[i*3+2] = mesh.rest_pos[i](2)*1.5f;
        }
        set_positions(mesh, init_pos);
        zero_velocities(mesh);

        float init_d = max_displacement(mesh, rp);
        printf("  Initial max displacement: %.4f\n", init_d);
        float max_ke = 0.0f, max_disp = init_d;
        for (int s = 0; s < 200; ++s) {
            solver.step(mesh, sim_cons, pin_cons);
            max_ke   = std::max(max_ke,   kinetic_energy(mesh));
            max_disp = std::max(max_disp, max_displacement(mesh, rp));
        }
        printf("  Final max disp: %.4f, max KE: %.4f\n",
               max_displacement(mesh, rp), max_ke);
        CHECK(max_ke   < 5000.0f);
        CHECK(max_disp < init_d * 3.0f);
    }

    // Test 5: Bent initial state + strong bend → energy stays bounded
    SECTION("Test 5: Bent initial + strong bend constraints → bounded energy");
    {
        ClothMesh mesh;
        generate_square_cloth(10, 10, 0.1f, 0, mesh);
        mesh.precompute_rest_state(0.1f);

        MeshTopology topo = MeshTopology::build(mesh);
        SimConstraints sim_cons;
        // bend_k must be << stretch_k for Jacobi-PD convergence (see DESIGN.md)
        sim_cons.build_stretch(mesh, 10.0f);
        sim_cons.build_bend(mesh, topo, 0.5f);
        mesh.upload_to_gpu();
        sim_cons.upload_to_gpu();
        sim_cons.precompute_jacobi_diag(mesh, 0.01f);

        Constraints pin_cons;
        pin_cons.pin_corners(mesh, 10);
        pin_cons.upload_to_gpu();

        PDSolverConfig cfg;
        cfg.dt = 0.01f; cfg.gravity = 0.0f; cfg.max_iterations = 200;
        PDSolver solver(cfg, mesh, sim_cons);

        auto rp = rest_flat(mesh);
        std::vector<float> init_pos(mesh.num_verts * 3);
        const float R = 0.5f;
        for (int i = 0; i < 10; ++i)
            for (int j = 0; j < 10; ++j) {
                int idx = i*10+j;
                float th = (i-4.5f)*0.1f/R;
                init_pos[idx*3+0] = mesh.rest_pos[idx](0);
                init_pos[idx*3+1] = R*(1.0f-std::cos(th));
                init_pos[idx*3+2] = R*std::sin(th);
            }
        set_positions(mesh, init_pos);
        zero_velocities(mesh);

        float init_d = max_displacement(mesh, rp);
        printf("  Initial max displacement: %.4f\n", init_d);
        float max_ke = 0.0f, max_disp = init_d;
        for (int s = 0; s < 200; ++s) {
            solver.step(mesh, sim_cons, pin_cons);
            max_ke   = std::max(max_ke,   kinetic_energy(mesh));
            max_disp = std::max(max_disp, max_displacement(mesh, rp));
        }
        printf("  Final max disp: %.4f, max KE: %.4f\n",
               max_displacement(mesh, rp), max_ke);
        CHECK(max_ke   < 500.0f);
        CHECK(max_disp < init_d * 3.0f);
    }

    printf("\n");
    return test_summary();
}
