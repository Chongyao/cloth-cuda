#include "cloth_mesh.h"
#include "mesh_topology.h"
#include "sim_constraints.h"
#include "constraints.h"
#include "mesh_generator.h"
#include "pd_solver.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>

#ifdef CUDA_MS_HAVE_CUDA
#include <cuda_runtime.h>
#endif

// Export mesh positions to ASCII PLY
static void export_ply(const ClothMesh& mesh, const char* filename)
{
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    fprintf(fp, "ply\nformat ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", mesh.num_verts);
    fprintf(fp, "property float x\nproperty float y\nproperty float z\n");
    fprintf(fp, "element face %d\n", mesh.num_tris);
    fprintf(fp, "property list uchar int vertex_indices\nend_header\n");

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
    for (int i = 0; i < mesh.num_verts; ++i)
        fprintf(fp, "%f %f %f\n", h_pos[i*3+0], h_pos[i*3+1], h_pos[i*3+2]);
    for (const auto& tri : mesh.triangles)
        fprintf(fp, "3 %d %d %d\n", tri(0), tri(1), tri(2));
    fclose(fp);
}

// Compute total (kinetic + gravitational potential) energy
static float compute_energy(const ClothMesh& mesh, const PDSolverConfig& cfg)
{
    std::vector<float> h_pos(mesh.num_verts * 3);
    std::vector<float> h_vel(mesh.num_verts * 3);
    std::vector<float> h_mass(mesh.num_verts);
#ifdef CUDA_MS_HAVE_CUDA
    cudaMemcpy(h_pos.data(),  mesh.d_pos,  mesh.num_verts * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel.data(),  mesh.d_vel,  mesh.num_verts * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mass.data(), mesh.d_mass, mesh.num_verts     * sizeof(float), cudaMemcpyDeviceToHost);
#else
    for (int i = 0; i < mesh.num_verts; ++i) {
        h_pos[i*3+0] = mesh.rest_pos[i](0);
        h_pos[i*3+1] = mesh.rest_pos[i](1);
        h_pos[i*3+2] = mesh.rest_pos[i](2);
        h_vel[i*3+0] = h_vel[i*3+1] = h_vel[i*3+2] = 0.0f;
        h_mass[i] = mesh.mass[i];
    }
#endif
    float kinetic = 0.0f, potential = 0.0f;
    for (int i = 0; i < mesh.num_verts; ++i) {
        const float m  = h_mass[i];
        const float vx = h_vel[i*3+0], vy = h_vel[i*3+1], vz = h_vel[i*3+2];
        kinetic   += 0.5f * m * (vx*vx + vy*vy + vz*vz);
        potential += m * (-cfg.gravity) * h_pos[i*3+1];
    }
    return kinetic + potential;
}

static void print_usage(const char* prog)
{
    fprintf(stderr, "Usage: %s <nrows> <ncols> <size> [options]\n\n", prog);
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  --steps <n>      Simulation steps        (default: 100)\n");
    fprintf(stderr, "  --dt <t>         Time step               (default: 0.01)\n");
    fprintf(stderr, "  --pin <mode>     top | corners           (default: top)\n");
    fprintf(stderr, "  --stiffness <k>  Stretch stiffness       (default: 1.0)\n");
    fprintf(stderr, "  --bend <k>       Bend stiffness          (default: 0 = off)\n");
    fprintf(stderr, "  --iter <n>       PD iterations/frame     (default: 50)\n");
    fprintf(stderr, "  --type <n>       Mesh type 0-3           (default: 3)\n");
    fprintf(stderr, "  --gravity <g>    Gravitational accel     (default: -9.8)\n");
    fprintf(stderr, "  --damping <d>    Velocity damping 0-1    (default: 0)\n");
    fprintf(stderr, "  --export <dir>   Export PLY frames\n");
    fprintf(stderr, "  --verbose        Per-frame energy output\n");
}

int main(int argc, char* argv[])
{
    if (argc < 4) { print_usage(argv[0]); return 1; }

    const int   nrows = std::stoi(argv[1]);
    const int   ncols = std::stoi(argv[2]);
    const float size  = std::stof(argv[3]);

    // Parse options
    int         steps          = 100;
    float       dt             = 0.01f;
    std::string pin_mode       = "top";
    float       stiffness      = 1.0f;
    float       bend_stiffness = 0.0f;
    int         iterations     = 50;
    int         mesh_type      = 3;
    float       gravity        = -9.8f;
    float       damping        = 0.0f;
    std::string export_dir;
    bool        verbose        = false;

    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--steps"     && i+1 < argc) steps          = std::stoi(argv[++i]);
        else if (a == "--dt"        && i+1 < argc) dt             = std::stof(argv[++i]);
        else if (a == "--pin"       && i+1 < argc) pin_mode       = argv[++i];
        else if (a == "--stiffness" && i+1 < argc) stiffness      = std::stof(argv[++i]);
        else if (a == "--bend"      && i+1 < argc) bend_stiffness = std::stof(argv[++i]);
        else if (a == "--iter"      && i+1 < argc) iterations     = std::stoi(argv[++i]);
        else if (a == "--type"      && i+1 < argc) mesh_type      = std::stoi(argv[++i]);
        else if (a == "--gravity"   && i+1 < argc) gravity        = std::stof(argv[++i]);
        else if (a == "--damping"   && i+1 < argc) damping        = std::stof(argv[++i]);
        else if (a == "--export"    && i+1 < argc) export_dir     = argv[++i];
        else if (a == "--verbose")                 verbose        = true;
    }

    printf("=== PD Cloth Simulation ===\n");
    printf("Grid: %d x %d, size: %.4f, type: %d\n", nrows, ncols, size, mesh_type);
    printf("Steps: %d, dt: %.4f, iter: %d\n", steps, dt, iterations);
    printf("stiffness: %.3f, bend: %.4f, damping: %.3f, pin: %s\n",
           stiffness, bend_stiffness, damping, pin_mode.c_str());

    // ---- Build mesh ----
    ClothMesh mesh;
    generate_square_cloth(nrows, ncols, size, mesh_type, mesh);
    mesh.precompute_rest_state(0.1f);

    // ---- Build topology (needed for bend constraints) ----
    printf("\n");
    MeshTopology topo = MeshTopology::build(mesh);

    // ---- Build sim constraints ----
    SimConstraints sim_cons;
    sim_cons.build_stretch(mesh, stiffness);
    if (bend_stiffness > 0.0f)
        sim_cons.build_bend(mesh, topo, bend_stiffness);

    printf("\nMesh: %d vertices, %d triangles, %d bend constraints\n",
           mesh.num_verts, mesh.num_tris, sim_cons.num_bend_cons);

    // ---- Setup pinned constraints ----
    Constraints pin_cons;
    if (pin_mode == "top")
        pin_cons.pin_top_row(mesh, ncols);
    else if (pin_mode == "corners")
        pin_cons.pin_corners(mesh, ncols);
    printf("Pinned vertices: %d\n", (int)pin_cons.pinned_indices.size());

    // ---- Upload to GPU ----
    mesh.upload_to_gpu();
    sim_cons.upload_to_gpu();
    sim_cons.precompute_jacobi_diag(mesh, dt);
    pin_cons.upload_to_gpu();

    // ---- Setup solver ----
    PDSolverConfig config;
    config.dt             = dt;
    config.max_iterations = iterations;
    config.stretch_stiffness = stiffness;
    config.bend_stiffness = bend_stiffness;
    config.gravity        = gravity;
    config.damping        = damping;
    config.use_chebyshev  = true;

    PDSolver solver(config, mesh, sim_cons);

    // ---- Initial export ----
    if (!export_dir.empty()) {
        char fn[256];
        snprintf(fn, sizeof(fn), "%s/frame_0000.ply", export_dir.c_str());
        export_ply(mesh, fn);
    }

    // ---- Simulation loop ----
    printf("\nSimulating...\n");
    printf("Initial energy: %.6f\n", compute_energy(mesh, config));

    const int export_interval = std::max(1, steps / 10);

    for (int step = 1; step <= steps; ++step) {
        solver.step(mesh, sim_cons, pin_cons);

        if (verbose && step % 10 == 0)
            printf("Step %d: Energy = %.6f\n", step, compute_energy(mesh, config));

        if (!export_dir.empty() && step % export_interval == 0) {
            char fn[256];
            snprintf(fn, sizeof(fn), "%s/frame_%04d.ply",
                     export_dir.c_str(), step / export_interval);
            export_ply(mesh, fn);
        }
    }

    printf("Final energy: %.6f\n", compute_energy(mesh, config));

    if (!export_dir.empty()) {
        char fn[256];
        snprintf(fn, sizeof(fn), "%s/frame_final.ply", export_dir.c_str());
        export_ply(mesh, fn);
        printf("\nExported frames to %s/\n", export_dir.c_str());
    }

    printf("\nSimulation complete.\n");
    return 0;
}
