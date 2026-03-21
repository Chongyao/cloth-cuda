#include "mesh.h"
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
    int   nrows = std::stoi(argv[1]);
    int   ncols = std::stoi(argv[2]);
    float size  = std::stof(argv[3]);
    int   type  = (argc >= 5) ? std::stoi(argv[4]) : 0;

    ClothMesh mesh;
    generate_square_cloth(nrows, ncols, size, type, mesh);
    mesh.precompute_rest_state(0.1f);
    mesh.build_inner_edges();
    mesh.print_stats();
    return EXIT_SUCCESS;
}
