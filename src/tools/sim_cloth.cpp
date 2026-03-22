#include "mesh.h"
#include "mesh_generator.h"
#include "constraints.h"
#include "pd_solver.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>

#ifdef CUDA_MS_HAVE_CUDA
#include <cuda_runtime.h>
#endif

// Export mesh to PLY format
void export_ply(const ClothMesh& mesh, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    fprintf(fp, "ply\n");
    fprintf(fp, "format ascii 1.0\n");
    fprintf(fp, "element vertex %d\n", mesh.num_verts);
    fprintf(fp, "property float x\n");
    fprintf(fp, "property float y\n");
    fprintf(fp, "property float z\n");
    fprintf(fp, "element face %d\n", mesh.num_tris);
    fprintf(fp, "property list uchar int vertex_indices\n");
    fprintf(fp, "end_header\n");

    // Read positions from GPU
    std::vector<float> h_pos(mesh.num_verts * 3);
#ifdef CUDA_MS_HAVE_CUDA
    cudaMemcpy(h_pos.data(), mesh.d_pos, mesh.num_verts * 3 * sizeof(float), cudaMemcpyDeviceToHost);
#else
    // CPU fallback
    for (int i = 0; i < mesh.num_verts; ++i) {
        h_pos[i * 3 + 0] = mesh.rest_pos[i](0);
        h_pos[i * 3 + 1] = mesh.rest_pos[i](1);
        h_pos[i * 3 + 2] = mesh.rest_pos[i](2);
    }
#endif

    for (int i = 0; i < mesh.num_verts; ++i) {
        fprintf(fp, "%f %f %f\n",
                h_pos[i * 3 + 0],
                h_pos[i * 3 + 1],
                h_pos[i * 3 + 2]);
    }

    for (const auto& tri : mesh.triangles) {
        fprintf(fp, "3 %d %d %d\n", tri(0), tri(1), tri(2));
    }

    fclose(fp);
}

// Compute total energy
float compute_energy(const ClothMesh& mesh, const PDSolverConfig& config) {
    std::vector<float> h_pos(mesh.num_verts * 3);
    std::vector<float> h_vel(mesh.num_verts * 3);
    std::vector<float> h_mass(mesh.num_verts);

#ifdef CUDA_MS_HAVE_CUDA
    cudaMemcpy(h_pos.data(), mesh.d_pos, mesh.num_verts * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel.data(), mesh.d_vel, mesh.num_verts * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mass.data(), mesh.d_mass, mesh.num_verts * sizeof(float), cudaMemcpyDeviceToHost);
#else
    for (int i = 0; i < mesh.num_verts; ++i) {
        h_pos[i * 3 + 0] = mesh.rest_pos[i](0);
        h_pos[i * 3 + 1] = mesh.rest_pos[i](1);
        h_pos[i * 3 + 2] = mesh.rest_pos[i](2);
        h_vel[i * 3 + 0] = h_vel[i * 3 + 1] = h_vel[i * 3 + 2] = 0.0f;
        h_mass[i] = mesh.mass[i];
    }
#endif

    float kinetic = 0.0f;
    float potential = 0.0f;

    for (int i = 0; i < mesh.num_verts; ++i) {
        float m = h_mass[i];
        float vx = h_vel[i * 3 + 0];
        float vy = h_vel[i * 3 + 1];
        float vz = h_vel[i * 3 + 2];

        kinetic += 0.5f * m * (vx * vx + vy * vy + vz * vz);
        potential += m * (-config.gravity) * h_pos[i * 3 + 1];  // y is up
    }

    return kinetic + potential;
}

void print_usage(const char* prog) {
    fprintf(stderr, "Usage: %s <nrows> <ncols> <size> [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --steps <n>        Number of simulation steps (default: 100)\n");
    fprintf(stderr, "  --dt <t>           Time step (default: 0.01)\n");
    fprintf(stderr, "  --pin <mode>       Constraint mode: top, corners (default: top)\n");
    fprintf(stderr, "  --stiffness <k>    Stretch stiffness (default: 1.0)\n");
    fprintf(stderr, "  --iter <n>         PD iterations per frame (default: 50)\n");
    fprintf(stderr, "  --type <n>         Mesh type: 0=uniform\\, 1=checker, 2=uniform/, 3=米字格 (default: 3)\n");
    fprintf(stderr, "  --bend <k>         Bend stiffness (default: 0 = disabled)\n");
    fprintf(stderr, "                     Jacobi-PD converges for k << stretch; recommend k <= 0.05\n");
    fprintf(stderr, "  --gravity <g>      Gravity acceleration (default: -9.8)\n");
    fprintf(stderr, "  --damping <d>      Velocity damping 0-1 (default: 0 = no damping)\n");
    fprintf(stderr, "  --export <dir>     Export frames to directory\n");
    fprintf(stderr, "  --verbose          Print per-frame statistics\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s 20 20 0.05 --steps 1000 --pin top\n", prog);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    int nrows = std::stoi(argv[1]);
    int ncols = std::stoi(argv[2]);
    float size = std::stof(argv[3]);

    // Default parameters
    int steps = 100;
    float dt = 0.01f;
    std::string pin_mode = "top";
    float stiffness = 1.0f;
    int iterations = 50;
    int mesh_type = 3;
    float bend_stiffness = 0.0f;
    float gravity = -9.8f;
    float damping = 0.0f;
    std::string export_dir;
    bool verbose = false;

    // Parse options
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) {
            steps = std::stoi(argv[++i]);
        } else if (arg == "--dt" && i + 1 < argc) {
            dt = std::stof(argv[++i]);
        } else if (arg == "--pin" && i + 1 < argc) {
            pin_mode = argv[++i];
        } else if (arg == "--stiffness" && i + 1 < argc) {
            stiffness = std::stof(argv[++i]);
        } else if (arg == "--iter" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--type" && i + 1 < argc) {
            mesh_type = std::stoi(argv[++i]);
        } else if (arg == "--bend" && i + 1 < argc) {
            bend_stiffness = std::stof(argv[++i]);
        } else if (arg == "--gravity" && i + 1 < argc) {
            gravity = std::stof(argv[++i]);
        } else if (arg == "--damping" && i + 1 < argc) {
            damping = std::stof(argv[++i]);
        } else if (arg == "--export" && i + 1 < argc) {
            export_dir = argv[++i];
        } else if (arg == "--verbose") {
            verbose = true;
        }
    }

    printf("=== PD Cloth Simulation ===\n");
    printf("Grid: %d x %d, Cell size: %f\n", nrows, ncols, size);
    printf("Steps: %d, dt: %f, Iterations: %d\n", steps, dt, iterations);
    printf("Stiffness: %f, Bend: %f, Damping: %f, Pin: %s, Mesh type: %d\n",
           stiffness, bend_stiffness, damping, pin_mode.c_str(), mesh_type);

    // Generate mesh
    ClothMesh mesh;
    generate_square_cloth(nrows, ncols, size, mesh_type, mesh);
    mesh.precompute_rest_state(0.1f);

    // Build topology and constraints
    printf("\n=== Building Topology ===\n");
    mesh.build_topology();
    mesh.build_stretch_from_topo(stiffness);
    if (bend_stiffness > 0.0f)
        mesh.build_bend_from_topo(bend_stiffness);

    printf("\nMesh: %d vertices, %d triangles, %d stretch, %d bend constraints\n",
           mesh.num_verts, mesh.num_tris, mesh.num_stretch_cons, mesh.num_bend_cons);

    // Setup constraints
    Constraints cons;
    if (pin_mode == "top") {
        cons.pin_top_row(mesh, ncols);
    } else if (pin_mode == "corners") {
        cons.pin_corners(mesh, ncols);
    }
    printf("Pinned vertices: %d\n", (int)cons.pinned_indices.size());

    // Upload to GPU
    mesh.upload_to_gpu();
    cons.upload_to_gpu();

    // Precompute Jacobi diagonal
    mesh.precompute_jacobi_diag(dt, stiffness);

    // Setup solver
    PDSolverConfig config;
    config.dt = dt;
    config.max_iterations = iterations;
    config.stretch_stiffness = stiffness;
    config.gravity = gravity;
    config.damping = damping;
    config.use_chebyshev = true;

    PDSolver solver(config, mesh);

    // Initial export
    if (!export_dir.empty()) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/frame_0000.ply", export_dir.c_str());
        export_ply(mesh, filename);
    }

    // Simulation loop
    printf("\nSimulating...\n");
    float total_energy = compute_energy(mesh, config);
    printf("Initial energy: %.6f\n", total_energy);

    int export_interval = steps / 10;  // Export 10 frames
    if (export_interval < 1) export_interval = 1;

    for (int step = 1; step <= steps; ++step) {
        solver.step(mesh, cons);

        if (verbose && step % 10 == 0) {
            float energy = compute_energy(mesh, config);
            printf("Step %d: Energy = %.6f\n", step, energy);
        }

        if (!export_dir.empty() && step % export_interval == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/frame_%04d.ply",
                     export_dir.c_str(), step / export_interval);
            export_ply(mesh, filename);
        }
    }

    total_energy = compute_energy(mesh, config);
    printf("Final energy: %.6f\n", total_energy);

    // Final export
    if (!export_dir.empty()) {
        char filename[256];
        snprintf(filename, sizeof(filename), "%s/frame_final.ply", export_dir.c_str());
        export_ply(mesh, filename);
        printf("\nExported frames to %s/\n", export_dir.c_str());
    }

    printf("\nSimulation complete.\n");
    return 0;
}
