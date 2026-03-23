#include "cloth_mesh.h"
#include "mesh_topology.h"
#include "mesh_generator.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// Usage: gen_cloth <nrows> <ncols> <size> [type]
int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <nrows> <ncols> <size> [type]\n", argv[0]);
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

    return EXIT_SUCCESS;
}
