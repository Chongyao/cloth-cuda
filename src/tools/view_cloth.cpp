#include "viewer/viewer.h"
#include "cloth_mesh.h"
#include "mesh_topology.h"
#include "mesh_generator.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// Usage: view_cloth <nrows> <ncols> <size> [type]
int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <nrows> <ncols> <size> [type]\n", argv[0]);
        fprintf(stderr, "  type: 0=uniform \\  1=checkerboard  2=uniform /  3=米字格 (default: 0)\n");
        return EXIT_FAILURE;
    }

    const int   nrows = std::stoi(argv[1]);
    const int   ncols = std::stoi(argv[2]);
    const float size  = std::stof(argv[3]);
    const int   type  = (argc >= 5) ? std::stoi(argv[4]) : 0;

    ClothMesh mesh;
    generate_square_cloth(nrows, ncols, size, type, mesh);
    mesh.precompute_rest_state(0.1f);
    mesh.print_stats();

    MeshTopology topo = MeshTopology::build(mesh);
    printf("  Inner (bend) edges: %d\n", topo.num_inner_edges());

    ClothViewer viewer;
    if (!viewer.init(1280, 720, "cuda-ms viewer")) {
        fprintf(stderr, "Failed to initialize viewer\n");
        return EXIT_FAILURE;
    }

    viewer.upload_mesh(mesh);

    while (!viewer.should_close()) {
        viewer.begin_frame();
        viewer.render(mesh);
        viewer.end_frame();
    }

    return EXIT_SUCCESS;
}
