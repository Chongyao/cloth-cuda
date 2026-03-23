#pragma once
#include "cloth_mesh.h"

// Generate a regular square cloth mesh.
//
// nrows, ncols : number of vertex rows and columns
// size         : edge length of each grid cell
// type         : triangulation pattern
//                  0 = uniform diagonal  \ (top-right to bottom-left)
//                  1 = checkerboard (alternating \ and /)
//                  2 = uniform diagonal  / (top-left to bottom-right)
//                  3 = 米字格 (cross pattern: center vertex per cell, 4 tris)
//                      isotropic — no directional bias, recommended for PD
void generate_square_cloth(int nrows, int ncols, float size, int type,
                           ClothMesh& mesh);
