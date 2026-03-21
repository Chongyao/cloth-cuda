#include "viewer/viewer.h"
#include "mesh.h"
#include "mesh_generator.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// Usage: view_cloth <nrows> <ncols> <size> [type]
int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <nrows> <ncols> <size> [type]\n", argv[0]);
        fprintf(stderr, "  nrows, ncols: number of vertex rows and columns\n");
        fprintf(stderr, "  size: edge length of each grid cell\n");
        fprintf(stderr, "  type: triangulation pattern (0=uniform \\, 1=checkerboard, 2=uniform /)\n");
        return EXIT_FAILURE;
    }

    int   nrows = std::stoi(argv[1]);
    int   ncols = std::stoi(argv[2]);
    float size  = std::stof(argv[3]);
    int   type  = (argc >= 5) ? std::stoi(argv[4]) : 0;

    // Generate mesh
    ClothMesh mesh;
    generate_square_cloth(nrows, ncols, size, type, mesh);
    mesh.precompute_rest_state(0.1f);
    mesh.build_inner_edges();
    mesh.print_stats();

    // Initialize viewer
    ClothViewer viewer;
    if (!viewer.init(1280, 720, "cuda-ms viewer")) {
        fprintf(stderr, "Failed to initialize viewer\n");
        return EXIT_FAILURE;
    }

    viewer.upload_mesh(mesh);

    // Main loop
    while (!viewer.should_close()) {
        viewer.begin_frame();
        viewer.render(mesh);
        viewer.end_frame();
    }

    return EXIT_SUCCESS;
}
