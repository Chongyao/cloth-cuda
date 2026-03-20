#include "obj_io.h"

#include <fstream>
#include <sstream>
#include <cstdio>

bool load_obj(const std::string& path, ObjMesh& out) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "load_obj: cannot open '%s'\n", path.c_str());
        return false;
    }

    out.positions.clear();
    out.indices.clear();

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;

        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            out.positions.push_back(x);
            out.positions.push_back(y);
            out.positions.push_back(z);
        } else if (token == "f") {
            std::vector<int> verts;
            std::string ftoken;
            while (ss >> ftoken) {
                int v = 0;
                sscanf(ftoken.c_str(), "%d", &v);
                verts.push_back(v - 1);
            }
            for (int i = 1; i + 1 < (int)verts.size(); ++i) {
                out.indices.push_back(verts[0]);
                out.indices.push_back(verts[i]);
                out.indices.push_back(verts[i + 1]);
            }
        }
    }

    if (out.positions.empty()) {
        fprintf(stderr, "load_obj: no vertices in '%s'\n", path.c_str());
        return false;
    }
    if (out.indices.empty()) {
        fprintf(stderr, "load_obj: no faces in '%s'\n", path.c_str());
        return false;
    }
    return true;
}
