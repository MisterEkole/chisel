// src/reconstruction/meshing.cpp
#include "reconstruction/meshing.h"

#include <Eigen/Dense>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <numeric>
#include <queue>

namespace chisel {
namespace reconstruction {

// ─── KNN helper (brute force for now) ───────
static std::vector<int> find_knn(const std::vector<Vec3>& points,
                                  int query_idx, int k) {
    struct Dist { int idx; double d; };
    std::vector<Dist> dists;
    dists.reserve(points.size());
    const Vec3& q = points[query_idx];
    for (size_t i = 0; i < points.size(); ++i) {
        if ((int)i == query_idx) continue;
        dists.push_back({(int)i, (q - points[i]).squaredNorm()});
    }
    std::partial_sort(dists.begin(),
                      dists.begin() + std::min(k, (int)dists.size()),
                      dists.end(),
                      [](const Dist& a, const Dist& b) { return a.d < b.d; });

    std::vector<int> result;
    for (int i = 0; i < std::min(k, (int)dists.size()); ++i)
        result.push_back(dists[i].idx);
    return result;
}

// ─── Normal estimation via PCA ──────────────
std::vector<Vec3> estimate_normals(const std::vector<Vec3>& points,
                                    int knn, float radius) {
    std::vector<Vec3> normals(points.size(), Vec3::Zero());

    for (size_t i = 0; i < points.size(); ++i) {
        auto neighbors = find_knn(points, i, knn);
        if (neighbors.size() < 3) continue;

        // Compute centroid
        Vec3 centroid = Vec3::Zero();
        for (int ni : neighbors) centroid += points[ni];
        centroid /= neighbors.size();

        // Covariance matrix
        Mat3 cov = Mat3::Zero();
        for (int ni : neighbors) {
            Vec3 d = points[ni] - centroid;
            cov += d * d.transpose();
        }

        // Smallest eigenvector = normal
        Eigen::SelfAdjointEigenSolver<Mat3> solver(cov);
        normals[i] = solver.eigenvectors().col(0);

        // Consistent orientation: point towards centroid of scene
        // (simple heuristic; a proper method would use MST)
        if (normals[i].dot(Vec3(0, 0, 1)) < 0)
            normals[i] = -normals[i];
    }

    std::cout << "[mesh] Estimated normals for " << points.size() << " points\n";
    return normals;
}

// ─── Simple Delaunay-based meshing ──────────
static TriangleMesh mesh_delaunay(const std::vector<Vec3>& points,
                                   const std::vector<Vec3>& colors,
                                   const MeshingConfig& cfg) {
    TriangleMesh mesh;
    mesh.vertices = points;
    mesh.vertex_colors = colors;

    // Build triangles by connecting K nearest neighbors
    // This is a simplified approach; a full implementation would use
    // 3D Delaunay tetrahedralization → surface extraction
    int knn = 12;
    double max_edge2 = cfg.max_edge_length * cfg.max_edge_length;

    for (size_t i = 0; i < points.size(); ++i) {
        auto neighbors = find_knn(points, i, knn);

        for (size_t j = 0; j < neighbors.size(); ++j) {
            for (size_t k = j + 1; k < neighbors.size(); ++k) {
                int ni = neighbors[j], nk = neighbors[k];

                // Check all edge lengths
                double d1 = (points[i] - points[ni]).squaredNorm();
                double d2 = (points[i] - points[nk]).squaredNorm();
                double d3 = (points[ni] - points[nk]).squaredNorm();

                if (d1 > max_edge2 || d2 > max_edge2 || d3 > max_edge2)
                    continue;

                // Only add if i < ni < nk to avoid duplicates
                if ((int)i < ni && ni < nk)
                    mesh.triangles.push_back(Eigen::Vector3i(i, ni, nk));
            }
        }
    }

    mesh.compute_normals();
    std::cout << "[mesh] Delaunay: " << mesh.num_vertices() << " vertices, "
              << mesh.num_triangles() << " triangles\n";
    return mesh;
}

// ─── Main meshing entry point ───────────────
TriangleMesh generate_mesh(const std::vector<Vec3>& points,
                            const std::vector<Vec3>& colors,
                            const std::vector<Vec3>& normals,
                            const MeshingConfig& cfg) {
    switch (cfg.method) {
        case MeshingConfig::Method::DELAUNAY:
            return mesh_delaunay(points, colors, cfg);
        case MeshingConfig::Method::POISSON:
            // TODO: implement Poisson surface reconstruction
            std::cout << "[mesh] Poisson not yet implemented, "
                         "falling back to Delaunay\n";
            return mesh_delaunay(points, colors, cfg);
        case MeshingConfig::Method::BALL_PIVOT:
            std::cout << "[mesh] Ball pivot not yet implemented, "
                         "falling back to Delaunay\n";
            return mesh_delaunay(points, colors, cfg);
    }
    return {};
}

// ─── Remove small components (BFS) ──────────
void remove_small_components(TriangleMesh& mesh, int min_triangles) {
    if (mesh.triangles.empty()) return;

    int n = static_cast<int>(mesh.vertices.size());
    std::vector<std::vector<int>> adj(n);

    for (size_t t = 0; t < mesh.triangles.size(); ++t) {
        auto& tri = mesh.triangles[t];
        adj[tri[0]].push_back(tri[1]);
        adj[tri[0]].push_back(tri[2]);
        adj[tri[1]].push_back(tri[0]);
        adj[tri[1]].push_back(tri[2]);
        adj[tri[2]].push_back(tri[0]);
        adj[tri[2]].push_back(tri[1]);
    }

    std::vector<int> component(n, -1);
    int num_components = 0;

    for (int i = 0; i < n; ++i) {
        if (component[i] >= 0) continue;
        std::queue<int> q;
        q.push(i);
        component[i] = num_components;
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (int u : adj[v]) {
                if (component[u] < 0) {
                    component[u] = num_components;
                    q.push(u);
                }
            }
        }
        num_components++;
    }

    // Count triangles per component
    std::vector<int> tri_count(num_components, 0);
    for (auto& tri : mesh.triangles) {
        tri_count[component[tri[0]]]++;
    }

    // Keep only large components
    std::vector<bool> keep_vertex(n, false);
    for (int i = 0; i < n; ++i) {
        if (tri_count[component[i]] >= min_triangles)
            keep_vertex[i] = true;
    }

    // Rebuild mesh
    std::vector<int> new_idx(n, -1);
    TriangleMesh new_mesh;
    for (int i = 0; i < n; ++i) {
        if (keep_vertex[i]) {
            new_idx[i] = static_cast<int>(new_mesh.vertices.size());
            new_mesh.vertices.push_back(mesh.vertices[i]);
            if (i < (int)mesh.vertex_colors.size())
                new_mesh.vertex_colors.push_back(mesh.vertex_colors[i]);
        }
    }

    for (auto& tri : mesh.triangles) {
        if (new_idx[tri[0]] >= 0 && new_idx[tri[1]] >= 0 && new_idx[tri[2]] >= 0) {
            new_mesh.triangles.push_back(
                Eigen::Vector3i(new_idx[tri[0]], new_idx[tri[1]], new_idx[tri[2]]));
        }
    }

    int removed = mesh.num_triangles() - new_mesh.num_triangles();
    mesh = std::move(new_mesh);
    std::cout << "[mesh] Removed " << removed << " triangles from small components\n";
}

// ─── Laplacian smoothing ────────────────────
void smooth_mesh(TriangleMesh& mesh, int iterations, double lambda) {
    int n = static_cast<int>(mesh.vertices.size());
    std::vector<std::vector<int>> adj(n);

    for (auto& tri : mesh.triangles) {
        adj[tri[0]].push_back(tri[1]); adj[tri[0]].push_back(tri[2]);
        adj[tri[1]].push_back(tri[0]); adj[tri[1]].push_back(tri[2]);
        adj[tri[2]].push_back(tri[0]); adj[tri[2]].push_back(tri[1]);
    }

    for (int iter = 0; iter < iterations; ++iter) {
        std::vector<Vec3> new_verts(n);
        for (int i = 0; i < n; ++i) {
            if (adj[i].empty()) { new_verts[i] = mesh.vertices[i]; continue; }
            Vec3 avg = Vec3::Zero();
            for (int j : adj[i]) avg += mesh.vertices[j];
            avg /= adj[i].size();
            new_verts[i] = mesh.vertices[i] + lambda * (avg - mesh.vertices[i]);
        }
        mesh.vertices = std::move(new_verts);
    }
}

// ─── PLY export ─────────────────────────────
bool write_mesh_ply(const std::string& path, const TriangleMesh& mesh) {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    bool has_colors = mesh.vertex_colors.size() == mesh.vertices.size();
    bool has_normals = mesh.vertex_normals.size() == mesh.vertices.size();

    out << "ply\nformat ascii 1.0\n"
        << "element vertex " << mesh.num_vertices() << "\n"
        << "property float x\nproperty float y\nproperty float z\n";
    if (has_normals)
        out << "property float nx\nproperty float ny\nproperty float nz\n";
    if (has_colors)
        out << "property uchar red\nproperty uchar green\nproperty uchar blue\n";
    out << "element face " << mesh.num_triangles() << "\n"
        << "property list uchar int vertex_indices\n"
        << "end_header\n";

    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        out << mesh.vertices[i].x() << " "
            << mesh.vertices[i].y() << " "
            << mesh.vertices[i].z();
        if (has_normals)
            out << " " << mesh.vertex_normals[i].x()
                << " " << mesh.vertex_normals[i].y()
                << " " << mesh.vertex_normals[i].z();
        if (has_colors)
            out << " " << static_cast<int>(mesh.vertex_colors[i].x() * 255)
                << " " << static_cast<int>(mesh.vertex_colors[i].y() * 255)
                << " " << static_cast<int>(mesh.vertex_colors[i].z() * 255);
        out << "\n";
    }

    for (auto& tri : mesh.triangles) {
        out << "3 " << tri[0] << " " << tri[1] << " " << tri[2] << "\n";
    }

    std::cout << "[mesh] Wrote PLY: " << path << "\n";
    return true;
}

// ─── OBJ export ─────────────────────────────
bool write_mesh_obj(const std::string& path, const TriangleMesh& mesh) {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    out << "# Chisel mesh export\n";
    for (auto& v : mesh.vertices) {
        out << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }
    for (auto& n : mesh.vertex_normals) {
        out << "vn " << n.x() << " " << n.y() << " " << n.z() << "\n";
    }
    for (auto& tri : mesh.triangles) {
        out << "f " << tri[0]+1 << " " << tri[1]+1 << " " << tri[2]+1 << "\n";
    }

    std::cout << "[mesh] Wrote OBJ: " << path << "\n";
    return true;
}

}  // namespace reconstruction
}  // namespace chisel
