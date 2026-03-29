#pragma once

#ifndef __CUDACC__
#  include "cloth_mesh.h"
#  include "sim_constraints.h"
#  include "constraints.h"
#  include <string>
#  include <vector>
#endif

enum class StretchBackend {
    GpuCurrent,
    CpuReference,
};

#ifndef __CUDACC__
// DiffCloth-aligned CPU reference triangle stretch solver.
// This path prioritizes correctness over performance and assembles the full
// stretch system on CPU using per-triangle dF/dx blocks.
struct CpuStretchReferenceSolver {
    explicit CpuStretchReferenceSolver(const ClothMesh& mesh,
                                       const SimConstraints& sim_cons,
                                       const Constraints& pin_cons,
                                       float dt,
                                       float gravity,
                                       float damping);

    void step(ClothMesh& mesh, const Constraints& pin_cons);
    static const char* backend_name() { return "cpu-ref"; }

private:
    int num_verts_ = 0;
    int num_tris_  = 0;
    float dt_      = 0.01f;
    float gravity_ = -9.8f;
    float damping_ = 0.0f;

    // Cached flattened geometry / reference data
    std::vector<float> masses_;
    std::vector<float> rest_area_;
    std::vector<float> stretch_k_;
    std::vector<int>   tris_;      // [T*3]
    std::vector<float> inv_duv_;   // [T*4] DiffCloth inv_deltaUV (col-major 2x2)
    std::vector<float> dFdx_;      // [T*54] 6x9 row-major per triangle

    void project_triangle_to_manifold(const float* x, int t, float* out_proj6) const;
};
#endif
