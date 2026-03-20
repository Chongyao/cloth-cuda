#pragma once

#include <string>
#include <vector>

struct ObjMesh {
    std::vector<float> positions;  // [v0x, v0y, v0z, v1x, ...]
    std::vector<int>   indices;    // [f0v0, f0v1, f0v2, f1v0, ...]

    int num_verts() const { return static_cast<int>(positions.size() / 3); }
    int num_tris()  const { return static_cast<int>(indices.size()   / 3); }
};

bool load_obj(const std::string& path, ObjMesh& out);
