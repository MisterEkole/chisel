// src/core/mesh_utils.cpp
#include "core/types.h"
#include <iostream>

namespace chisel {

void TriangleMesh::compute_normals() {
    vertex_normals.assign(vertices.size(), Vec3::Zero());

    for (auto& tri : triangles) {
        const Vec3& v0 = vertices[tri[0]];
        const Vec3& v1 = vertices[tri[1]];
        const Vec3& v2 = vertices[tri[2]];

        Vec3 normal = (v1 - v0).cross(v2 - v0);
        double area = normal.norm();
        if (area > 1e-12) normal /= area;  // unit normal

        // Area-weighted accumulation
        vertex_normals[tri[0]] += normal * area;
        vertex_normals[tri[1]] += normal * area;
        vertex_normals[tri[2]] += normal * area;
    }

    for (auto& n : vertex_normals) {
        double len = n.norm();
        if (len > 1e-12) n /= len;
    }

    std::cout << "[mesh] Computed normals for " << vertices.size()
              << " vertices\n";
}

}  // namespace chisel
