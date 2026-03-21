#pragma once
#include "mesh.h"

// Generate a regular square cloth mesh.
//
// nrows, ncols : number of vertex rows and columns
// size         : edge length of each grid cell
// type         : triangulation pattern
//                  0 = uniform diagonal  \ (top-right / bottom-left)
//                  1 = checkerboard (alternating \ and /)
//                  2 = uniform diagonal  / (top-left / bottom-right)
void generate_square_cloth(int nrows, int ncols, float size, int type,
                           ClothMesh& mesh);
